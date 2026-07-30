# AIUpscaleTweak

A LiveContainer dylib tweak that does one thing: real-time, on-device AI
upscaling of a Metal-rendered game, using Apple's **MetalFX** spatial
scaler (GPU/Neural Engine accelerated, purpose-built for exactly this).

## Why MetalFX and not a downloaded ONNX model

Worth knowing before you build this: a general super-resolution net
(Real-ESRGAN etc.) does maybe 1–5 fps on a phone — it's a photo tool, not
something you can bolt onto a 60fps game loop. MetalFX *is* a local,
on-device neural upscaler that Apple ships specifically to hit real-time
framerates on A-series/M-series GPUs, so it's the actual answer to "AI
model that runs well on iPhone 15/16 Pro and upscales a game live." It also
does edge-aware reconstruction as part of upscaling, which meaningfully
cleans up jaggies — think of it as most of the way to the antialiasing you
asked about. True temporal AA (MetalFX's `MTLFXTemporalScaler`, DLAA-style)
needs motion vectors, jitter, and depth from the game engine itself, which
you can't get from blind injection — that's the one piece genuinely out of
reach for a drop-in tweak. Spatial scaling is what's here.

## What it does

1. Hooks `CAMetalLayer.setDrawableSize:` so the game is told to render at
   ~72% of native resolution (tunable via `kRenderScale`) — a real GPU
   perf win, not just visual trickery.
2. Hooks the drawable/command-buffer present path so every frame, right
   before it would hit the screen, the low-res texture is run through
   `MTLFXSpatialScaler` back up to native resolution and shown via an
   overlay layer. The game's own (now hidden) low-res layer never displays.

## Build

Requires macOS + Xcode (iOS SDK isn't available in a plain shell):

```
chmod +x build.sh
./build.sh
```

Produces `AIUpscaleTweak.dylib`, ad-hoc signed. None of the APIs used here
are new to iOS 27 — Metal, CAMetalLayer, and MetalFX have all been stable
since iOS 16 — so building against whatever current stable Xcode/SDK you
have is enough; the result runs fine on an iOS 27 device, no beta SDK
required.

## Install (iOS 27 beta 2 note)

Per LiveContainer's tweak docs (livecontainer.github.io/docs/guides/tweaks):
drop `AIUpscaleTweak.dylib` into LiveContainer's **Tweaks** folder — either
globally, or scoped to one specific app if you only want it on one game.
LiveContainer re-signs and injects it automatically on next launch.

iOS 27 changed dyld/shared-cache internals enough to break LiveContainer's
original JIT-bypass mechanism. That's since been fixed upstream, but two
things matter for you right now:

- **Update LiveContainer to the latest release** — iOS 27 beta support
  landed after the initial break, older builds won't launch guest apps
  correctly.
- **Leave "Spoof SDK Version" disabled.** It's still broken on iOS 27 and
  will crash guest apps on launch if enabled.

This tweak itself doesn't depend on either of those internals — it hooks
Metal/CAMetalLayer at the Objective-C runtime level, which is unrelated to
LiveContainer's own JIT-bypass patching — but it obviously can't do
anything if the guest app isn't launching in the first place, so get
LiveContainer itself working before troubleshooting this.

To disable without removing the file, launch with the environment variable
`AIU_UPSCALE_DISABLE=1` set (if your LiveContainer build exposes per-app
env vars in its tweak/launch settings).

## Tuning

Both constants are at the top of `AIUpscaleTweak.m`:

- `kRenderScale` — internal render resolution as a fraction of native.
  Lower = more headroom, more reconstruction artifacts. 0.6–0.8 is sane.
- `kMinLayerDimension` — layers smaller than this are left alone, so tiny
  UI-only Metal surfaces some apps use don't get hijacked.

## CI / Releases

`.github/workflows/build-and-release.yml` builds the dylib on a
GitHub-hosted macOS runner on every push/PR, and on any tag matching
`v*.*.*` also cuts a GitHub Release with `AIUpscaleTweak.dylib` attached.

```
git tag v1.0.0
git push origin v1.0.0
```

That's it — the release appears with the built dylib as a downloadable
asset. No secrets or signing certs needed since the build uses ad-hoc
codesigning, same as `build.sh` locally.

One caveat: GitHub's hosted macOS runners carry whatever Xcode versions
they've published as stable, which typically lags the newest iOS beta SDK
by a bit. As noted above, that's not actually a problem here since nothing
in this code needs the iOS 27 SDK specifically — but if you ever add an
iOS 27-only API and CI can't find it, that's why; you'd need a self-hosted
runner on a Mac with the beta Xcode installed for that case.

## Real limitations, not hedging

- Only touches **Metal**-rendered apps. Very old titles still on OpenGL
  ES/GLKit won't be affected — this is the vast majority of anything from
  roughly the last decade, but not literally everything old.
- Assumes one primary full-screen Metal layer, true for essentially all
  games, but a multi-surface app would only get the first one it finds.
- The drawable/command-buffer hook targets Apple-private classes,
  resolved at runtime by probing an instance rather than a hardcoded name
  (the standard way to do this without a jailbreak). It's runtime-verified
  each launch, but if Apple restructures Metal's internals in a future iOS
  version, worth re-checking that the probe still finds real classes.
- MetalFX symbol names here are written from documented API surface. I
  can't compile-check this without Xcode, so verify them against
  `MetalFX.h` in your SDK on first build in case of minor naming drift.
