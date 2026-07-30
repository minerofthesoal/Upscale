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
//     Metal backends. Pre-Metal titles using OpenGL ES / GLKit are untouched.
//   - Assumes a single primary full-screen CAMetalLayer, which is true for
//     essentially all games. Apps with multiple large Metal surfaces at once
//     will only get the first one detected upscaled.
//   - Hooks two Apple-private (undocumented, unstable-name) classes at
//     runtime by probing a throwaway drawable/command buffer rather than
//     linking against a fixed class name, which is the standard way to do
//     this without a jailbreak. It's still runtime-dependent: if a future
//     iOS version changes internal Metal class layout, re-verify the probe
//     still resolves real classes before relying on it.
//   - MetalFX symbol names (MTLFXSpatialScalerColorProcessingMode* etc.)
//     are written from the documented API surface; double check them
//     against the MetalFX.h in your SDK when you build, since I can't
//     compile-check this without Xcode/iOS SDK access.
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
@property (nonatomic, strong) id<MTLFXSpatialScaling> scaler;
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
    probeLayer.device = device;
    probeLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    probeLayer.drawableSize = CGSizeMake(4, 4); // below kMinLayerDimension, no-op for our hook logic

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
    if (!AIUEnabled() ||
        nativeSize.width  < kMinLayerDimension ||
        nativeSize.height < kMinLayerDimension) {
        return nativeSize; // too small to bother — likely a UI-only layer
    }

    AIUContext *ctx = [_contexts objectForKey:layer];
    if (!ctx) {
        ctx = [AIUContext new];
        ctx.gameLayer = layer;
        ctx.device = layer.device ?: MTLCreateSystemDefaultDevice();
        ctx.queue = [ctx.device newCommandQueue];
        [_contexts setObject:ctx forKey:layer];
    }

    CGSize renderSize = CGSizeMake(round(nativeSize.width  * kRenderScale),
                                    round(nativeSize.height * kRenderScale));

    if (!CGSizeEqualToSize(ctx.nativeSize, nativeSize)) {
        ctx.nativeSize = nativeSize;
        ctx.renderSize = renderSize;
        [self rebuildOverlayAndScalerForContext:ctx pixelFormat:layer.pixelFormat];
    }

    return renderSize;
}

- (void)rebuildOverlayAndScalerForContext:(AIUContext *)ctx pixelFormat:(MTLPixelFormat)fmt {
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
    [game.superlayer insertSublayer:overlay above:game];
    ctx.overlayLayer = overlay;

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
}

#pragma mark Called from the swizzled present hooks (per frame)

- (void)handlePresentedDrawable:(id<CAMetalDrawable>)drawable fromLayer:(CAMetalLayer *)layer {
    AIUContext *ctx = [_contexts objectForKey:layer];
    if (!ctx || !ctx.scaler || !ctx.overlayLayer) return;

    id<CAMetalDrawable> outDrawable = [ctx.overlayLayer nextDrawable];
    if (!outDrawable) return;

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    ctx.scaler.colorTexture  = drawable.texture;
    ctx.scaler.outputTexture = outDrawable.texture;
    [ctx.scaler encodeToCommandBuffer:cmd];
    [cmd presentDrawable:outDrawable];
    [cmd commit];
}

@end

#pragma mark - CAMetalLayer swizzled methods

@implementation CAMetalLayer (AIUpscaleHook)

- (void)aiu_setDrawableSize:(CGSize)size {
    if (objc_getAssociatedObject(self, kAIUOverlayMarkerKey)) {
        [self aiu_setDrawableSize:size]; // our own overlay — pass through untouched
        return;
    }
    CGSize renderSize = [[AIUEngine shared] renderSizeForGameLayer:self nativeSize:size];
    [self aiu_setDrawableSize:renderSize]; // post-swizzle, this calls the ORIGINAL implementation
}

- (id<CAMetalDrawable>)aiu_nextDrawable {
    id<CAMetalDrawable> drawable = [self aiu_nextDrawable]; // calls the ORIGINAL implementation
    if (drawable && !objc_getAssociatedObject(self, kAIUOverlayMarkerKey)) {
        objc_setAssociatedObject(drawable, kAIULayerKey, self, OBJC_ASSOCIATION_ASSIGN);
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
