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

        ctx.scaler.colorTexture  = drawable.texture;
        ctx.scaler.outputTexture = outDrawable.texture;
        [ctx.scaler encodeToCommandBuffer:cmd];
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
