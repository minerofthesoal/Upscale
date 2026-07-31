//
//  AIUpscaleTweak.m
//
//  LiveContainer tweak: real-time on-device AI upscaling for Metal-based
//  iOS games, using Apple's MetalFX spatial scaler (GPU/Neural Engine
//  accelerated). One job: force a lower internal render resolution, then
//  reconstruct back to native res every frame before it hits the screen.
//
//  Install: drop the compiled .dylib into LiveContainer's Tweaks folder
//  (global, or per-app) per https://livecontainer.github.io/docs/guides/tweaks
//
//  Scope / limitations (read before filing a "doesn't work"):
//   - Only affects apps that render through CAMetalLayer (Metal). Covers the
//     vast majority of games from the last ~decade, including Unity/Unreal
//     Metal backends. Pre-Metal titles using OpenGL ES / GLKit are untouched
//     — e.g. Minecraft: Story Mode runs on Telltale's own "Telltale Tool"
//     engine from 2015 and was never updated after Telltale lost the
//     license, so it almost certainly never creates a CAMetalLayer at all.
//     For a game like that, this tweak should end up being a complete
//     no-op — there's nothing Metal-based in it to intercept, upscaled or
//     otherwise.
//   - Only ever locks onto ONE CAMetalLayer per process, and only if its
//     pixel size is close to the physical screen size (see
//     AIULooksLikeGameLayer). This is deliberately strict: iOS itself
//     creates CAMetalLayers internally for system compositing (blur
//     effects, UIKit/SwiftUI's own accelerated paths, etc.), and those can
//     also be large. Touching one of those — resizing it or inserting a
//     sibling layer into its hierarchy — has nothing to do with the game
//     and is a good way to crash the whole process before gameplay even
//     starts, which is the most likely explanation if you saw an instant
//     crash rather than a crash during actual play.
//   - Update: the instant crash reported on Minecraft: Story Mode turned
//     out to be unrelated to that game, or to which CAMetalLayer gets
//     targeted — it was a recursive dispatch_once inside this file's own
//     +[AIUEngine shared] singleton, triggered by the probe layer used to
//     discover private Metal classes during setup (see installHooks). It
//     fired on the very first load of the dylib, before any app code runs,
//     which is why it looked identical across apps. Fixed by tagging the
//     probe layer as "ours" before touching its drawableSize, same as the
//     overlay layer already was.
//   - Every hook body is still wrapped defensively and falls back to
//     passthrough behavior on failure rather than propagating, since a
//     single bad frame should never be able to take down the host app.
//     That said, @try/@catch only catches Objective-C NSExceptions — it
//     can't save you from a true low-level memory fault or a libdispatch
//     API-misuse crash like the one above, so "defensive" reduces risk,
//     it doesn't eliminate it.
//   - Hooks two Apple-private (undocumented, unstable-name) classes at
//     runtime by probing a throwaway drawable/command buffer rather than
//     linking against a fixed class name, which is the standard way to do
//     this without a jailbreak. It's still runtime-dependent: if a future
//     iOS version changes internal Metal class layout, re-verify the probe
//     still resolves real classes before relying on it.
//   - MetalFX symbol names have been verified against Apple's actual
//     public API surface (MTLFXSpatialScaler is a protocol, not
//     MTLFXSpatialScaling — fixed after an earlier build caught this).
//   - No MSAA. MSAA has to be baked into the game's own render pipeline —
//     its pipeline-state and attachment textures are created with a fixed
//     sample count before this tweak ever sees a frame, and we only ever
//     see the finished frame right before present. Forcing a sample count
//     from outside would mismatch the game's own attachments and hit
//     Metal's command validation, which is a good way to crash. What IS
//     included instead is a real FXAA pass (see kAIUFXAAShaderSource)
//     applied after the MetalFX upscale and before present — genuine
//     edge-detection antialiasing on top of MetalFX's own reconstruction,
//     without needing anything from the game's internals.
//

#import <objc/runtime.h>
#import <objc/message.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

#pragma mark - Tunables

// Fraction of native resolution the game is told to render at internally.
// Lower = more GPU headroom / battery savings, more work for MetalFX to
// reconstruct. 0.60–0.80 is a reasonable range on A19 Pro (iPhone 17 Pro)
// for 60fps titles — its 6-core GPU has a Neural Accelerator per core plus
// a 16-core Neural Engine, so there's real headroom above this range too.
static const CGFloat kRenderScale = 0.72;

// Run an FXAA post-process pass after the MetalFX upscale, before present.
// Adds one extra fullscreen render pass per frame — cheap on A19 Pro.
static const BOOL kEnableFXAA = YES;

// Ignore any CAMetalLayer smaller than this on either axis — avoids
// hijacking small UI-only Metal layers some apps use for effects.
static const CGFloat kMinLayerDimension = 240.0;

#pragma mark - Runtime env toggle (no recompile needed to disable)

static BOOL AIUEnabled(void) {
    static BOOL enabled;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        enabled = getenv("AIU_UPSCALE_DISABLE") == NULL;
    });
    return enabled;
}

#pragma mark - Associated object keys

static void *kAIUOverlayMarkerKey = &kAIUOverlayMarkerKey;
static void *kAIULayerKey         = &kAIULayerKey;

#pragma mark - Game-layer identification

// Only treat a layer as the game's primary render surface if its pixel
// size plausibly matches the physical screen. Deliberately stricter than a
// bare minimum-size check — see the header comment on why a system
// compositing layer must never be touched here.
static BOOL AIULooksLikeGameLayer(CGSize nativeSize) {
    if (nativeSize.width < kMinLayerDimension || nativeSize.height < kMinLayerDimension) {
        return NO;
    }
    CGSize screen = UIScreen.mainScreen.nativeBounds.size; // already in pixels
    if (screen.width <= 0 || screen.height <= 0) return NO;

    CGFloat wRatio = nativeSize.width  / screen.width;
    CGFloat hRatio = nativeSize.height / screen.height;
    // Slack for safe-area exclusion, split view, etc., but require both
    // axes to be close to full screen — rules out e.g. a large-but-square
    // internal thumbnail/effect layer that happens to pass a flat minimum.
    BOOL widthOK  = wRatio >= 0.5 && wRatio <= 1.05;
    BOOL heightOK = hRatio >= 0.5 && hRatio <= 1.05;
    return widthOK && heightOK;
}

#pragma mark - FXAA shader source (compiled at runtime, no .metallib needed)

// Standard luma-based FXAA: edge detection via a 3x3 luma neighborhood,
// then a directional blend along the detected edge, plus a subpixel term
// for thin/aliased detail. Runs as a single fullscreen-triangle pass.
//
// Orientation note: this flips V to match a top-left texture origin. If
// the AA'd image comes out upside down on your build, delete the
// "out.uv.y = 1.0 - out.uv.y;" line below — that's the only
// device/driver-dependent bit here and I can't verify it without a device.
static NSString *const kAIUFXAAShaderSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct AIUFXAAVertexOut {\n"
"    float4 position [[position]];\n"
"    float2 uv;\n"
"};\n"
"\n"
"vertex AIUFXAAVertexOut aiu_fxaa_vertex(uint vertexID [[vertex_id]]) {\n"
"    float2 positions[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };\n"
"    AIUFXAAVertexOut out;\n"
"    out.position = float4(positions[vertexID], 0.0, 1.0);\n"
"    out.uv = out.position.xy * 0.5 + 0.5;\n"
"    out.uv.y = 1.0 - out.uv.y;\n"
"    return out;\n"
"}\n"
"\n"
"inline float aiu_luma(float3 c) {\n"
"    return dot(c, float3(0.299, 0.587, 0.114));\n"
"}\n"
"\n"
"fragment float4 aiu_fxaa_fragment(AIUFXAAVertexOut in [[stage_in]],\n"
"                                   texture2d<float, access::sample> src [[texture(0)]],\n"
"                                   sampler smp [[sampler(0)]]) {\n"
"    constexpr float kEdgeThresholdMin = 0.0312;\n"
"    constexpr float kEdgeThreshold = 0.125;\n"
"    constexpr float kSubpixQuality = 0.75;\n"
"\n"
"    float2 texel = 1.0 / float2(src.get_width(), src.get_height());\n"
"    float2 uv = in.uv;\n"
"\n"
"    float3 rgbCenter = src.sample(smp, uv).rgb;\n"
"    float3 rgbN = src.sample(smp, uv + float2(0.0, -texel.y)).rgb;\n"
"    float3 rgbS = src.sample(smp, uv + float2(0.0,  texel.y)).rgb;\n"
"    float3 rgbE = src.sample(smp, uv + float2( texel.x, 0.0)).rgb;\n"
"    float3 rgbW = src.sample(smp, uv + float2(-texel.x, 0.0)).rgb;\n"
"\n"
"    float lumaCenter = aiu_luma(rgbCenter);\n"
"    float lumaN = aiu_luma(rgbN);\n"
"    float lumaS = aiu_luma(rgbS);\n"
"    float lumaE = aiu_luma(rgbE);\n"
"    float lumaW = aiu_luma(rgbW);\n"
"\n"
"    float lumaMin = min(lumaCenter, min(min(lumaN, lumaS), min(lumaE, lumaW)));\n"
"    float lumaMax = max(lumaCenter, max(max(lumaN, lumaS), max(lumaE, lumaW)));\n"
"    float lumaRange = lumaMax - lumaMin;\n"
"\n"
"    if (lumaRange < max(kEdgeThresholdMin, lumaMax * kEdgeThreshold)) {\n"
"        return float4(rgbCenter, 1.0);\n"
"    }\n"
"\n"
"    float3 rgbNE = src.sample(smp, uv + float2( texel.x, -texel.y)).rgb;\n"
"    float3 rgbNW = src.sample(smp, uv + float2(-texel.x, -texel.y)).rgb;\n"
"    float3 rgbSE = src.sample(smp, uv + float2( texel.x,  texel.y)).rgb;\n"
"    float3 rgbSW = src.sample(smp, uv + float2(-texel.x,  texel.y)).rgb;\n"
"\n"
"    float lumaNE = aiu_luma(rgbNE);\n"
"    float lumaNW = aiu_luma(rgbNW);\n"
"    float lumaSE = aiu_luma(rgbSE);\n"
"    float lumaSW = aiu_luma(rgbSW);\n"
"\n"
"    float edgeHorizontal = abs(lumaNW + lumaNE - 2.0 * lumaN) * 2.0\n"
"                          + abs(lumaW  + lumaE  - 2.0 * lumaCenter)\n"
"                          + abs(lumaSW + lumaSE - 2.0 * lumaS);\n"
"    float edgeVertical   = abs(lumaNW + lumaSW - 2.0 * lumaW) * 2.0\n"
"                          + abs(lumaN  + lumaS  - 2.0 * lumaCenter)\n"
"                          + abs(lumaNE + lumaSE - 2.0 * lumaE);\n"
"    bool isHorizontal = edgeHorizontal >= edgeVertical;\n"
"\n"
"    float luma1 = isHorizontal ? lumaN : lumaW;\n"
"    float luma2 = isHorizontal ? lumaS : lumaE;\n"
"    float gradient1 = luma1 - lumaCenter;\n"
"    float gradient2 = luma2 - lumaCenter;\n"
"    bool is1Steepest = abs(gradient1) >= abs(gradient2);\n"
"    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));\n"
"\n"
"    float stepLength = isHorizontal ? texel.y : texel.x;\n"
"    float lumaLocalAverage = 0.0;\n"
"    if (is1Steepest) {\n"
"        stepLength = -stepLength;\n"
"        lumaLocalAverage = 0.5 * (luma1 + lumaCenter);\n"
"    } else {\n"
"        lumaLocalAverage = 0.5 * (luma2 + lumaCenter);\n"
"    }\n"
"\n"
"    float2 currentUV = uv;\n"
"    if (isHorizontal) {\n"
"        currentUV.y += stepLength * 0.5;\n"
"    } else {\n"
"        currentUV.x += stepLength * 0.5;\n"
"    }\n"
"\n"
"    float2 offset = isHorizontal ? float2(texel.x, 0.0) : float2(0.0, texel.y);\n"
"    float2 uv1 = currentUV - offset;\n"
"    float2 uv2 = currentUV + offset;\n"
"\n"
"    float lumaEnd1 = aiu_luma(src.sample(smp, uv1).rgb) - lumaLocalAverage;\n"
"    float lumaEnd2 = aiu_luma(src.sample(smp, uv2).rgb) - lumaLocalAverage;\n"
"    bool reached1 = abs(lumaEnd1) >= gradientScaled;\n"
"    bool reached2 = abs(lumaEnd2) >= gradientScaled;\n"
"    bool reachedBoth = reached1 && reached2;\n"
"\n"
"    if (!reached1) uv1 -= offset;\n"
"    if (!reached2) uv2 += offset;\n"
"\n"
"    if (!reachedBoth) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            if (!reached1) {\n"
"                lumaEnd1 = aiu_luma(src.sample(smp, uv1).rgb) - lumaLocalAverage;\n"
"            }\n"
"            if (!reached2) {\n"
"                lumaEnd2 = aiu_luma(src.sample(smp, uv2).rgb) - lumaLocalAverage;\n"
"            }\n"
"            reached1 = abs(lumaEnd1) >= gradientScaled;\n"
"            reached2 = abs(lumaEnd2) >= gradientScaled;\n"
"            reachedBoth = reached1 && reached2;\n"
"            if (!reached1) uv1 -= offset;\n"
"            if (!reached2) uv2 += offset;\n"
"            if (reachedBoth) break;\n"
"        }\n"
"    }\n"
"\n"
"    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);\n"
"    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);\n"
"    bool isDirection1 = distance1 < distance2;\n"
"    float distanceFinal = min(distance1, distance2);\n"
"    float edgeThickness = (distance1 + distance2);\n"
"    float pixelOffset = -distanceFinal / edgeThickness + 0.5;\n"
"\n"
"    bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;\n"
"    bool correctVariation = ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;\n"
"    float finalOffset = correctVariation ? pixelOffset : 0.0;\n"
"\n"
"    float lumaAverage = (1.0/12.0) * (2.0*(lumaN+lumaS+lumaE+lumaW) + lumaNE+lumaNW+lumaSE+lumaSW);\n"
"    float subPixelOffset1 = clamp(abs(lumaAverage - lumaCenter) / lumaRange, 0.0, 1.0);\n"
"    float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;\n"
"    float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * kSubpixQuality;\n"
"    finalOffset = max(finalOffset, subPixelOffsetFinal);\n"
"\n"
"    float2 finalUV = uv;\n"
"    if (isHorizontal) {\n"
"        finalUV.y += finalOffset * stepLength;\n"
"    } else {\n"
"        finalUV.x += finalOffset * stepLength;\n"
"    }\n"
"\n"
"    float3 finalColor = src.sample(smp, finalUV).rgb;\n"
"    return float4(finalColor, 1.0);\n"
"}\n";

#pragma mark - Swizzle helper (for known, compile-time classes)

static void AIUSwizzleInstanceMethod(Class cls, SEL original, SEL replacement) {
    Method origMethod = class_getInstanceMethod(cls, original);
    Method newMethod  = class_getInstanceMethod(cls, replacement);
    if (!origMethod || !newMethod) return;
    BOOL added = class_addMethod(cls, original,
                                  method_getImplementation(newMethod),
                                  method_getTypeEncoding(newMethod));
    if (added) {
        class_replaceMethod(cls, replacement,
                             method_getImplementation(origMethod),
                             method_getTypeEncoding(origMethod));
    } else {
        method_exchangeImplementations(origMethod, newMethod);
    }
}

#pragma mark - Per-game-layer context

@interface AIUContext : NSObject
@property (nonatomic, weak)   CAMetalLayer *gameLayer;
@property (nonatomic, strong) CAMetalLayer *overlayLayer;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLFXSpatialScaler> scaler;
@property (nonatomic, assign) CGSize nativeSize;
@property (nonatomic, assign) CGSize renderSize;
// FXAA post-process pass (see kAIUFXAAShaderSource). upscaledIntermediate
// holds the native-res MetalFX output before FXAA reads it and writes the
// final antialiased image to the actual drawable.
@property (nonatomic, strong) id<MTLTexture> upscaledIntermediate;
@property (nonatomic, strong) id<MTLRenderPipelineState> fxaaPipeline;
@property (nonatomic, strong) id<MTLSamplerState> fxaaSampler;
@end
@implementation AIUContext @end

#pragma mark - Engine

@interface AIUEngine : NSObject
+ (instancetype)shared;
- (CGSize)renderSizeForGameLayer:(CAMetalLayer *)layer nativeSize:(CGSize)nativeSize;
- (void)handlePresentedDrawable:(id<CAMetalDrawable>)drawable fromLayer:(CAMetalLayer *)layer;
@end

// C trampolines used to hook the privately-named drawable / command-buffer
// classes discovered at runtime (can't write an @implementation category
// against a class whose name we don't know at compile time).
typedef void (*AIUPresentIMP)(id, SEL);
typedef void (*AIUPresentAtTimeIMP)(id, SEL, CFTimeInterval);
typedef void (*AIUPresentDrawableIMP)(id, SEL, id<MTLDrawable>);

static AIUPresentIMP         AIUOrigPresent         = NULL;
static AIUPresentAtTimeIMP   AIUOrigPresentAtTime   = NULL;
static AIUPresentDrawableIMP AIUOrigPresentDrawable = NULL;

static void AIUHookedPresent(id self_, SEL _cmd) {
    CAMetalLayer *layer = objc_getAssociatedObject(self_, kAIULayerKey);
    if (layer) [[AIUEngine shared] handlePresentedDrawable:self_ fromLayer:layer];
    if (AIUOrigPresent) AIUOrigPresent(self_, _cmd);
}

static void AIUHookedPresentAtTime(id self_, SEL _cmd, CFTimeInterval t) {
    CAMetalLayer *layer = objc_getAssociatedObject(self_, kAIULayerKey);
    if (layer) [[AIUEngine shared] handlePresentedDrawable:self_ fromLayer:layer];
    if (AIUOrigPresentAtTime) AIUOrigPresentAtTime(self_, _cmd, t);
}

static void AIUHookedPresentDrawable(id self_, SEL _cmd, id<MTLDrawable> drawable) {
    CAMetalLayer *layer = objc_getAssociatedObject(drawable, kAIULayerKey);
    if (layer && [drawable conformsToProtocol:@protocol(CAMetalDrawable)]) {
        [[AIUEngine shared] handlePresentedDrawable:(id<CAMetalDrawable>)drawable fromLayer:layer];
    }
    if (AIUOrigPresentDrawable) AIUOrigPresentDrawable(self_, _cmd, drawable);
}

@implementation AIUEngine {
    NSMapTable<CAMetalLayer *, AIUContext *> *_contexts;
    __weak CAMetalLayer *_lockedLayer; // once set, every other layer is left alone entirely
}

+ (instancetype)shared {
    static AIUEngine *s;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ s = [AIUEngine new]; });
    return s;
}

- (instancetype)init {
    if ((self = [super init])) {
        _contexts = [NSMapTable weakToStrongObjectsMapTable];
        @try {
            [self installHooks];
        } @catch (NSException *e) {
            NSLog(@"[AIUpscale] hook install failed, running inert: %@", e);
        }
    }
    return self;
}

#pragma mark Hook installation

- (void)installHooks {
    if (!AIUEnabled()) return;

    // 1. Force lower internal render resolution on the game's layer.
    AIUSwizzleInstanceMethod(CAMetalLayer.class,
                              @selector(setDrawableSize:),
                              @selector(aiu_setDrawableSize:));

    // 2. Tag each drawable with the layer it came from, so the present
    //    hooks below (which fire on private classes) know which game
    //    surface to upscale.
    AIUSwizzleInstanceMethod(CAMetalLayer.class,
                              @selector(nextDrawable),
                              @selector(aiu_nextDrawable));

    // 3. Discover the concrete private classes backing MTLCommandBuffer and
    //    CAMetalDrawable via a throwaway offscreen probe, then hook their
    //    presentation entry points with saved-IMP trampolines.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return;

    CAMetalLayer *probeLayer = [CAMetalLayer layer];
    // Mark this BEFORE touching drawableSize below. Without this, setting
    // drawableSize on our own probe layer routes through our just-installed
    // aiu_setDrawableSize: swizzle, which calls +[AIUEngine shared] again
    // while the first call's dispatch_once block (this very function, called
    // from -init) is still running on this thread. libdispatch detects that
    // exact reentrancy and crashes on purpose ("trying to lock recursively")
    // rather than deadlocking silently — this was the actual cause of the
    // instant crash on every app, not anything game-specific.
    objc_setAssociatedObject(probeLayer, kAIUOverlayMarkerKey, @YES, OBJC_ASSOCIATION_RETAIN);
    probeLayer.device = device;
    probeLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    probeLayer.drawableSize = CGSizeMake(4, 4); // now safely bypasses our hook logic

    id<CAMetalDrawable> probeDrawable = [probeLayer nextDrawable];
    id<MTLCommandQueue> probeQueue = [device newCommandQueue];
    id<MTLCommandBuffer> probeBuffer = [probeQueue commandBuffer];
    if (!probeDrawable || !probeBuffer) return;

    Class drawableClass = object_getClass(probeDrawable);
    Class commandBufferClass = object_getClass(probeBuffer);

    Method m1 = class_getInstanceMethod(drawableClass, @selector(present));
    if (m1) AIUOrigPresent = (AIUPresentIMP)method_setImplementation(m1, (IMP)AIUHookedPresent);

    Method m2 = class_getInstanceMethod(drawableClass, @selector(presentAtTime:));
    if (m2) AIUOrigPresentAtTime = (AIUPresentAtTimeIMP)method_setImplementation(m2, (IMP)AIUHookedPresentAtTime);

    Method m3 = class_getInstanceMethod(commandBufferClass, @selector(presentDrawable:));
    if (m3) AIUOrigPresentDrawable = (AIUPresentDrawableIMP)method_setImplementation(m3, (IMP)AIUHookedPresentDrawable);

    // Clean up the probe drawable (now routes through our real hook, which
    // no-ops safely since probeLayer never got a real context).
    [probeDrawable present];
}

#pragma mark Called from the swizzled setDrawableSize:

- (CGSize)renderSizeForGameLayer:(CAMetalLayer *)layer nativeSize:(CGSize)nativeSize {
    @try {
        if (!AIUEnabled()) return nativeSize;
        if (_lockedLayer && _lockedLayer != layer) return nativeSize; // already committed to a different layer
        if (!AIULooksLikeGameLayer(nativeSize)) return nativeSize;

        AIUContext *ctx = [_contexts objectForKey:layer];
        if (!ctx) {
            id<MTLDevice> device = layer.device ?: MTLCreateSystemDefaultDevice();
            if (!device) return nativeSize;
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) return nativeSize;

            ctx = [AIUContext new];
            ctx.gameLayer = layer;
            ctx.device = device;
            ctx.queue = queue;
            [_contexts setObject:ctx forKey:layer];
            _lockedLayer = layer;
        }

        CGSize renderSize = CGSizeMake(round(nativeSize.width  * kRenderScale),
                                        round(nativeSize.height * kRenderScale));

        if (!CGSizeEqualToSize(ctx.nativeSize, nativeSize)) {
            ctx.nativeSize = nativeSize;
            ctx.renderSize = renderSize;
            [self rebuildOverlayAndScalerForContext:ctx pixelFormat:layer.pixelFormat];
        }

        return renderSize;
    } @catch (NSException *e) {
        NSLog(@"[AIUpscale] renderSizeForGameLayer failed, passing size through untouched: %@", e);
        return nativeSize;
    }
}

- (void)rebuildOverlayAndScalerForContext:(AIUContext *)ctx pixelFormat:(MTLPixelFormat)fmt {
    @try {
        CAMetalLayer *game = ctx.gameLayer;
        if (!game) return;

        MTLPixelFormat pixelFormat = (fmt == MTLPixelFormatInvalid) ? MTLPixelFormatBGRA8Unorm : fmt;

        // Hide the low-res game layer — it's only used as an upscale source now.
        game.opacity = 0.0f;

        [ctx.overlayLayer removeFromSuperlayer];
        CAMetalLayer *overlay = [CAMetalLayer layer];
        objc_setAssociatedObject(overlay, kAIUOverlayMarkerKey, @YES, OBJC_ASSOCIATION_RETAIN);
        overlay.device = ctx.device;
        overlay.pixelFormat = pixelFormat;
        overlay.framebufferOnly = NO; // MetalFX needs shader-write access to the output texture
        overlay.frame = game.frame;
        overlay.contentsScale = game.contentsScale;
        overlay.drawableSize = ctx.nativeSize; // goes through the swizzle, but overlay marker bypasses hijack
        ctx.overlayLayer = overlay;

        // game.superlayer can legitimately be nil here if the game configures
        // its layer before adding it to a view hierarchy — sending
        // insertSublayer:above: to a nil superlayer is a safe no-op, and
        // handlePresentedDrawable:fromLayer: retries the attach opportunistically
        // on a later frame once a superlayer actually exists.
        [game.superlayer insertSublayer:overlay above:game];

        ctx.scaler = nil;
        if (@available(iOS 16.0, *)) {
            if ([MTLFXSpatialScalerDescriptor supportsDevice:ctx.device]) {
                MTLFXSpatialScalerDescriptor *desc = [MTLFXSpatialScalerDescriptor new];
                desc.inputWidth  = (NSUInteger)ctx.renderSize.width;
                desc.inputHeight = (NSUInteger)ctx.renderSize.height;
                desc.outputWidth  = (NSUInteger)ctx.nativeSize.width;
                desc.outputHeight = (NSUInteger)ctx.nativeSize.height;
                desc.colorTextureFormat = pixelFormat;
                desc.outputTextureFormat = pixelFormat;
                desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
                ctx.scaler = [desc newSpatialScalerWithDevice:ctx.device];
            } else {
                NSLog(@"[AIUpscale] device does not support MetalFX spatial scaling; running passthrough");
            }
        }

        // FXAA pass setup: an intermediate native-res texture for MetalFX
        // to write into, plus the compiled FXAA render pipeline that reads
        // it and writes the final antialiased image to the real drawable.
        // Any failure here just leaves ctx.fxaaPipeline nil, and
        // handlePresentedDrawable:fromLayer: falls back to writing the
        // MetalFX output straight to the drawable — upscale-only, no crash.
        ctx.upscaledIntermediate = nil;
        ctx.fxaaPipeline = nil;
        ctx.fxaaSampler = nil;
        if (kEnableFXAA && ctx.scaler) {
            MTLTextureDescriptor *td =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                     width:(NSUInteger)ctx.nativeSize.width
                                                                    height:(NSUInteger)ctx.nativeSize.height
                                                                 mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            td.storageMode = MTLStorageModePrivate;
            ctx.upscaledIntermediate = [ctx.device newTextureWithDescriptor:td];

            NSError *libErr = nil;
            id<MTLLibrary> lib = [ctx.device newLibraryWithSource:kAIUFXAAShaderSource
                                                            options:nil
                                                              error:&libErr];
            if (lib) {
                id<MTLFunction> vfn = [lib newFunctionWithName:@"aiu_fxaa_vertex"];
                id<MTLFunction> ffn = [lib newFunctionWithName:@"aiu_fxaa_fragment"];
                if (vfn && ffn) {
                    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
                    pd.vertexFunction = vfn;
                    pd.fragmentFunction = ffn;
                    pd.colorAttachments[0].pixelFormat = pixelFormat;
                    NSError *pipeErr = nil;
                    ctx.fxaaPipeline = [ctx.device newRenderPipelineStateWithDescriptor:pd error:&pipeErr];
                    if (!ctx.fxaaPipeline) {
                        NSLog(@"[AIUpscale] FXAA pipeline build failed, running upscale-only: %@", pipeErr);
                    }
                }
            } else {
                NSLog(@"[AIUpscale] FXAA shader compile failed, running upscale-only: %@", libErr);
            }

            MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
            sd.minFilter = MTLSamplerMinMagFilterLinear;
            sd.magFilter = MTLSamplerMinMagFilterLinear;
            sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            ctx.fxaaSampler = [ctx.device newSamplerStateWithDescriptor:sd];
        }
    } @catch (NSException *e) {
        NSLog(@"[AIUpscale] rebuildOverlayAndScalerForContext failed, disabling upscale for this layer: %@", e);
        ctx.scaler = nil;
        ctx.overlayLayer = nil;
    }
}

#pragma mark Called from the swizzled present hooks (per frame)

- (void)handlePresentedDrawable:(id<CAMetalDrawable>)drawable fromLayer:(CAMetalLayer *)layer {
    @try {
        AIUContext *ctx = [_contexts objectForKey:layer];
        if (!ctx || !ctx.scaler || !ctx.overlayLayer) return;

        // Opportunistic retry: if the overlay couldn't be attached when it
        // was built (game's layer had no superlayer yet at that point),
        // try again now that a frame is actually being presented.
        if (!ctx.overlayLayer.superlayer && layer.superlayer) {
            [layer.superlayer insertSublayer:ctx.overlayLayer above:layer];
        }
        if (!ctx.overlayLayer.superlayer) return;

        id<CAMetalDrawable> outDrawable = [ctx.overlayLayer nextDrawable];
        if (!outDrawable) return;

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        if (!cmd) return;

        BOOL useFXAA = kEnableFXAA && ctx.fxaaPipeline && ctx.fxaaSampler && ctx.upscaledIntermediate;

        // Pass 1: MetalFX spatial upscale. Writes straight to the drawable
        // if FXAA isn't available this frame, otherwise to the intermediate
        // texture that pass 2 will read from.
        ctx.scaler.colorTexture  = drawable.texture;
        ctx.scaler.outputTexture = useFXAA ? ctx.upscaledIntermediate : outDrawable.texture;
        [ctx.scaler encodeToCommandBuffer:cmd];

        // Pass 2: FXAA, intermediate -> drawable.
        if (useFXAA) {
            MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = outDrawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

            id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
            if (enc) {
                [enc setRenderPipelineState:ctx.fxaaPipeline];
                [enc setFragmentTexture:ctx.upscaledIntermediate atIndex:0];
                [enc setFragmentSamplerState:ctx.fxaaSampler atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [enc endEncoding];
            }
        }

        [cmd presentDrawable:outDrawable];
        [cmd commit];
    } @catch (NSException *e) {
        NSLog(@"[AIUpscale] handlePresentedDrawable failed for this frame, skipping: %@", e);
    }
}

@end

#pragma mark - CAMetalLayer swizzled methods

@implementation CAMetalLayer (AIUpscaleHook)

- (void)aiu_setDrawableSize:(CGSize)size {
    if (objc_getAssociatedObject(self, kAIUOverlayMarkerKey)) {
        [self aiu_setDrawableSize:size]; // our own overlay — pass through untouched
        return;
    }
    CGSize renderSize = size;
    @try {
        renderSize = [[AIUEngine shared] renderSizeForGameLayer:self nativeSize:size];
    } @catch (NSException *e) {
        NSLog(@"[AIUpscale] setDrawableSize hook failed, passing size through untouched: %@", e);
        renderSize = size;
    }
    [self aiu_setDrawableSize:renderSize]; // post-swizzle, this calls the ORIGINAL implementation
}

- (id<CAMetalDrawable>)aiu_nextDrawable {
    id<CAMetalDrawable> drawable = [self aiu_nextDrawable]; // calls the ORIGINAL implementation
    @try {
        if (drawable && !objc_getAssociatedObject(self, kAIUOverlayMarkerKey)) {
            objc_setAssociatedObject(drawable, kAIULayerKey, self, OBJC_ASSOCIATION_ASSIGN);
        }
    } @catch (NSException *e) {
        NSLog(@"[AIUpscale] nextDrawable tagging failed, drawable will present untouched: %@", e);
    }
    return drawable;
}

@end

#pragma mark - Entry point

__attribute__((constructor))
static void AIUpscaleTweakInit(void) {
    @autoreleasepool {
        [AIUEngine shared];
        NSLog(@"[AIUpscale] loaded — render scale %.2f, MetalFX upscale active on next Metal layer", kRenderScale);
    }
}
