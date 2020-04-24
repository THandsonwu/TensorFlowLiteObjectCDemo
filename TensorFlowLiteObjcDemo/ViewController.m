//
//  ViewController.m
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import "ViewController.h"

#import <AVFoundation/AVFoundation.h>
#import <TFLTensorFlowLite/TFLTensorFlowLite.h>
#import <Accelerate/Accelerate.h>
#import <libextobjc/extobjc.h>
#import "HFCameraHelper.h"
#import "OverlayView.h"
#import "HFInference.h"

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
//相机
@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) AVCaptureDevice *inputDevice;
@property (strong, nonatomic) AVCaptureDeviceInput *deviceInput;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, assign) AVCaptureVideoOrientation videoOrientation;
@property (nonatomic, strong) HFCameraHelper *helper;

//延时识别
@property (nonatomic, assign) NSTimeInterval previousTime;
@property (nonatomic, assign) NSTimeInterval delayBetweenMs;

//TensorFlow
@property (strong, nonatomic) TFLInterpreter *interpreter;

//绘图
@property (nonatomic, strong) OverlayView *overlayView;
@property (nonatomic, strong) CIContext *context;//自定义裁剪需要
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    //初始化一些参数
    self.previousTime = [[NSDate distantPast] timeIntervalSince1970] *1000;
    self.delayBetweenMs = 1000;
    
    //监听屏幕旋转,主要是锁定竖屏之后判断镜头取景方向
    [self.helper startUpdateAccelerometerResult:^(UIImageOrientation orientation) {
        switch (orientation) {
            case UIImageOrientationUp:
            {
                 NSLog(@"home在下");
                self.videoOrientation = AVCaptureVideoOrientationPortrait;
                break;
            }
            case UIImageOrientationDown:
            {
                NSLog(@"home在上");
                self.videoOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
                break;
            }
            case UIImageOrientationLeft:
            {
                NSLog(@"home在右");
                self.videoOrientation = AVCaptureVideoOrientationLandscapeRight;
                break;
            }
            case UIImageOrientationRight:
            {
                NSLog(@"home在左");
                self.videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
                break;
            }
            default:
                break;
        }
    }];
    [self setupInterpreter];
    [self setupCamera];
    
}


#pragma mark------ AVCaptureVideoDataOutputSampleBufferDelegate

- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    NSTimeInterval curentInterval = [[NSDate date] timeIntervalSince1970] * 1000;
    
    if (curentInterval - self.previousTime < self.delayBetweenMs) {
       return;
    }
    /*
    if (connection.videoOrientation != self.videoOrientation) {
        //切换镜头方向,如果是官方训练模型不必要切换,可以注释掉
        connection.videoOrientation = self.videoOrientation;
    }
     */
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    size_t imageWidth = CVPixelBufferGetWidth(pixelBuffer);
    size_t imageHeight = CVPixelBufferGetHeight(pixelBuffer);
   
    //如果需要旋转识别图像,可以用下面的方法,但是在iOS13.4上,内存释放有问题
    /*
    CVPixelBufferRef  rotatePixel = pixelBuffer;
    switch (self.videoOrientation) {
        case 1:
            rotatePixel = [self rotateBuffer:pixelBuffer withConstant:0];
            break;
        case 2:
            rotatePixel = [self rotateBuffer:pixelBuffer withConstant:2];
            break;
        case 3:
            rotatePixel = [self rotateBuffer:pixelBuffer withConstant:1];
            break;
        case 4:
            rotatePixel = [self rotateBuffer:pixelBuffer withConstant:3];
            break;
        default:
            break;
    }
     */
    
    //如果需要裁剪并且缩放识别图像,可以用下面方法,需要自己设定裁剪范围,并且计算仿射变换
    /*
    CGRect videoRect = CGRectMake(0, 0, imageWidth, imageHeight);
    CGSize scaledSize = CGSizeMake(300, 300);

    // Create a rectangle that meets the output size's aspect ratio, centered in the original video frame
    CGSize cropSize = CGSizeZero;
    if (imageWidth > imageHeight) {
        cropSize = CGSizeMake(imageWidth, imageWidth * 3 /4);
    }
    else
    {
        cropSize = CGSizeMake(imageWidth, imageWidth * 4 /3);
    }
    CGRect centerCroppingRect = AVMakeRectWithAspectRatioInsideRect(cropSize, videoRect);
    CVPixelBufferRef croppedAndScaled = [self createCroppedPixelBufferRef:pixelBuffer cropRect:centerCroppingRect scaleSize:scaledSize context:self.context];
    */
    //这里用的官方的训练模型,识别大小为300 * 300,所以直接缩放
    CVPixelBufferRef scaledPixelBuffer = [self resized:CGSizeMake(300, 300) cvpixelBuffer:pixelBuffer];
    //如果想看看缩放之后的图像是否满足要求,可以保存到相册
    /*
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        UIImage *image =  [self imageFromSampleBuffer:scaledPixelBuffer];
        UIImageWriteToSavedPhotosAlbum(image, self, @selector(image:didFinishSavingWithError:contextInfo:), (__bridge void *)self);
    });
     */
    
    //TensorFlow 输入和输出数据处理
    NSError *error;
    TFLTensor *inputTensor = [self.interpreter inputTensorAtIndex:0 error:&error];
    NSData *imageData = [self rgbDataFromBuffer:scaledPixelBuffer isModelQuantized:inputTensor.dataType == TFLTensorDataTypeUInt8];
    [inputTensor copyData:imageData error:&error];
    
    [self.interpreter invokeWithError:&error];
    if (error) {
        NSLog(@"Error++: %@", error);
    }
    //输出坐标,按照top,left,bottom,right的占比
    TFLTensor *outputTensor = [self.interpreter outputTensorAtIndex:0 error:&error];
    //输出index
    TFLTensor *outputClasses = [self.interpreter outputTensorAtIndex:1 error:nil];
    //输出分数
    TFLTensor *outputScores = [self.interpreter outputTensorAtIndex:2 error:nil];
    //输出识别物体个数
    TFLTensor *outputCount = [self.interpreter outputTensorAtIndex:3 error:nil];
    
    //格式化输出的数据
    NSArray<HFInference *> *inferences = [self formatTensorResultWith:[self transTFLTensorOutputData:outputTensor] indexs:[self transTFLTensorOutputData:outputClasses] scores:[self transTFLTensorOutputData:outputScores] count:[[self transTFLTensorOutputData:outputCount].firstObject integerValue] width:imageWidth height:imageHeight];
    
    NSLog(@"+++++++++++++");
    for (HFInference *inference in inferences) {
        NSLog(@"rect: %@  index %ld  score: %f className: %@\n",NSStringFromCGRect(inference.boundingRect),inference.index,inference.confidence,inference.className);
    }
    NSLog(@"+++++++++++++");
    //切换到主线程绘制
    dispatch_async(dispatch_get_main_queue(), ^{
        [self drawOverLayWithInferences:inferences width:imageWidth height:imageHeight];
    });
    
    
}

#pragma mark----TensorFlowLite

- (NSData *)rgbDataFromBuffer:(CVPixelBufferRef)pixelBuffer isModelQuantized:(BOOL)isQuantized
{
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    unsigned char* sourceData = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    if (!sourceData) {
        return nil;
    }
    size_t width =  CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    size_t sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    int destinationChannelCount = 3;
    size_t destinationBytesPerRow = destinationChannelCount * width;
    
    vImage_Buffer inbuff = {sourceData, height, width, sourceRowBytes};
   
    unsigned char *destinationData = malloc(height * destinationBytesPerRow);
    if (destinationData == nil) {
        return nil;
    }
    
    vImage_Buffer  outbuff = {destinationData,height,width,destinationBytesPerRow};
  
    if (CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA)
    {
        vImageConvert_BGRA8888toRGB888(&inbuff, &outbuff, kvImageNoFlags);
    }
    else if (CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32ARGB)
    {
        vImageConvert_ARGB8888toRGB888(&inbuff, &outbuff, kvImageNoFlags);
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    CVPixelBufferRelease(pixelBuffer);//记得释放资源
  
    NSData *data = [[NSData alloc] initWithBytes:outbuff.data length:outbuff.rowBytes *height];
    
    if (destinationData != NULL) {
          free(destinationData);
          destinationData = NULL;
    }
    
    if (isQuantized) {
        return  data;
    }
    
    Byte *bytesPtr = (Byte *)[data bytes];
    //针对不是量化模型,需要转换成float类型的数据
    NSMutableData *rgbData = [[NSMutableData alloc] initWithCapacity:0];

    for (int i = 0; i < data.length; i++) {
        Byte byte = (Byte)bytesPtr[i];
        float bytf = (float)byte / 255.0;
       [rgbData appendBytes:&bytf length:sizeof(float)];
    }
    
    return rgbData;
}

- (NSArray *)transTFLTensorOutputData:(TFLTensor *)outpuTensor
{
    NSMutableArray * arry = [NSMutableArray array];
    
    float output[40U];
    [[outpuTensor dataWithError:nil] getBytes:output length:(sizeof(float) *40U)];
    if ([outpuTensor.name isEqualToString:@"TFLite_Detection_PostProcess"]) {
        for (NSInteger i = 0; i < 10U; i++) {
            // top left bottom right
            UIEdgeInsets inset = UIEdgeInsetsMake(output[4* i + 0], output[4* i + 1], output[4* i + 2], output[4* i + 3]);
            [arry addObject:[NSValue valueWithUIEdgeInsets:inset]];
        }
    }
    else if ([outpuTensor.name isEqualToString:@"TFLite_Detection_PostProcess:1"] ||[outpuTensor.name isEqualToString:@"TFLite_Detection_PostProcess:2"])
    {
         for (NSInteger i = 0; i < 10U; i++) {
            [arry addObject:[NSNumber numberWithFloat:output[i]]];
        }
    }
    else if ([outpuTensor.name isEqualToString:@"TFLite_Detection_PostProcess:3"])
    {
//        NSNumber *count = output[0] ? [NSNumber numberWithFloat:output[0]] : [NSNumber numberWithFloat:0.0];
        NSNumber *count = @10;
        [arry addObject:count];
    }
    
    return arry;
}

- (NSArray<HFInference *> *)formatTensorResultWith:(NSArray *)outputBoundingBox indexs:(NSArray *)indexs scores:(NSArray *)scores count:(NSInteger)count width:(CGFloat)width height:(CGFloat)height
{
    NSMutableArray<HFInference *> *arry = [NSMutableArray arrayWithCapacity:count];
    
    for (NSInteger i = 0; i < count; i++) {
        CGFloat confidence = [scores[i] floatValue];
        
        if (confidence < 0.5) {
            continue;
        }
        NSInteger index = [indexs[i] integerValue] + 1;//官方模型需要+1;
        CGRect rect = CGRectZero;
        UIEdgeInsets inset;
        [outputBoundingBox[i] getValue:&inset];
        rect.origin.y = inset.top;
        rect.origin.x = inset.left;
        rect.size.height = inset.bottom - rect.origin.y;
        rect.size.width = inset.right - rect.origin.x;

        CGRect newRect = CGRectApplyAffineTransform(rect, CGAffineTransformMakeScale(width, height));
        //如果是自定义并且图片识别有方向的话,就用下面的方法
//        CGRect newRect = [self fixOriginSizeWithInset:inset videoOrientation:self.videoOrientation width:width height:height];
        HFInference *inference = [HFInference new];
        inference.confidence = confidence;
        inference.index = index;
        inference.boundingRect = newRect;
        inference.className = [self loadLabels:@"labelmap"][index];
        [arry addObject:inference];

    }
    
    return arry;
}

- (CGRect)fixOriginSizeWithInset:(UIEdgeInsets)inset videoOrientation:(AVCaptureVideoOrientation)orientation width:(CGFloat)width height:(CGFloat)height
{
    CGRect rect = CGRectZero;
    switch (orientation) {
        case AVCaptureVideoOrientationPortrait:
        {
            //已OK
            rect.origin.x = inset.left;
            rect.origin.y = inset.top ;
            rect.size.width = inset.right - inset.left;
            rect.size.height = (inset.bottom - inset.top);
            rect = CGRectApplyAffineTransform(rect, CGAffineTransformMakeScale(width, height));
        }
            break;
        case AVCaptureVideoOrientationLandscapeRight:
        {
            //已OK
            rect.origin.x = 1-inset.bottom;
            rect.origin.y =  inset.left;
            rect.size.width = inset.bottom - inset.top;
            rect.size.height = inset.right - inset.left;
            rect = CGRectApplyAffineTransform(rect, CGAffineTransformMakeScale(width, height));
        }
        
            break;
        case AVCaptureVideoOrientationLandscapeLeft:
        {
            //已OK
            rect.origin.x = inset.top;
            rect.origin.y = 1-inset.right;
            rect.size.width = inset.bottom - inset.top;
            rect.size.height = inset.right - inset.left;
            rect = CGRectApplyAffineTransform(rect, CGAffineTransformMakeScale(width, height));
        }
        
            break;
        case AVCaptureVideoOrientationPortraitUpsideDown:
        {
            //已OK
            rect.origin.x = 1 - inset.right;
            rect.origin.y = 1- inset.bottom ;
            rect.size.width = inset.right - inset.left;
            rect.size.height = inset.bottom - inset.top;
            rect = CGRectApplyAffineTransform(rect, CGAffineTransformMakeScale(width, height));
        }
        
            break;
        default:
            break;
    }
    return rect;
}

- (NSArray *)loadLabels:(NSString *)labelsInfo
{
    NSURL *path = [[NSBundle mainBundle] URLForResource:labelsInfo withExtension:@"txt"];
    if (path == nil) {
        return nil;
    }
    NSString *contents  = [NSString stringWithContentsOfURL:path encoding:NSUTF8StringEncoding error:nil];
    return [contents componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
}

#pragma mark----draw ObjectDetection
- (void)drawOverLayWithInferences:(NSArray<HFInference *> *)inferences width:(CGFloat)width height:(CGFloat)height
{
    [self.overlayView.overlays removeAllObjects];
    [self.overlayView setNeedsDisplay];
    
    if (inferences.count == 0) {
        return;
    }
    
    NSMutableArray<Overlayer *> * overlays = @[].mutableCopy;
    for (HFInference *inference in inferences) {
        CGRect convertedRect = CGRectApplyAffineTransform(inference.boundingRect , CGAffineTransformMakeScale(self.overlayView.bounds.size.width/width, self.overlayView.bounds.size.height / height));
        if (convertedRect.origin.x < 0) {
            convertedRect.origin.x = 5;
        }
        if (convertedRect.origin.y <0) {
            convertedRect.origin.y = 5;
        }
        
        if (CGRectGetMaxY(convertedRect) > CGRectGetMaxY(self.overlayView.bounds)) {
            convertedRect.size.height = CGRectGetMaxY(self.overlayView.bounds) - convertedRect.origin.y - 5;
        }
        
        if (CGRectGetMaxX(convertedRect) > CGRectGetMaxX(self.overlayView.bounds)) {
            convertedRect.size.width = CGRectGetMaxX(self.overlayView.bounds) - convertedRect.origin.x - 5;
        }
        
        Overlayer *layer = [Overlayer new];
        layer.borderRect = convertedRect;
        layer.color = UIColor.redColor;
        layer.name = [NSString stringWithFormat:@"%@ %.2f%%",inference.className,inference.confidence *100];
        NSDictionary *dic = @{NSFontAttributeName:[UIFont systemFontOfSize:14]};
        layer.nameStringSize = [layer.name boundingRectWithSize:CGSizeMake(MAXFLOAT, 20) options:(NSStringDrawingUsesLineFragmentOrigin) attributes:dic context:nil].size;
        layer.font = [UIFont systemFontOfSize:14];
        layer.nameDirection = self.videoOrientation;
        [overlays addObject:layer];
    }
    self.overlayView.overlays = overlays;
    [self.overlayView setNeedsDisplay];
}


#pragma mark---- pixelBuffer operation
//缩放CVPixelBufferRef
- (CVPixelBufferRef)resized:(CGSize)size cvpixelBuffer:(CVPixelBufferRef)pixelBuffer
{
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
//    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    size_t imageWidth =  CVPixelBufferGetWidth(pixelBuffer);
    size_t imageHeight = CVPixelBufferGetHeight(pixelBuffer);
    OSType pixelBufferType =  CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(pixelBufferType == kCVPixelFormatType_32BGRA);
    size_t sourceRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer);
    NSInteger imageChannels = 4;
   
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
 
    vImage_Buffer inbuff = {sourceBaseAddr, (NSUInteger)imageHeight,(NSUInteger)imageWidth, sourceRowBytes};
   
//    NSInteger scaledImageRowBytes = ceil(size.width/4) * 4  * imageChannels;
    NSInteger scaledImageRowBytes = vImageByteAlign(size.width * imageChannels , 64);
    
    unsigned char *scaledVImageBuffer = malloc((NSInteger)size.height * scaledImageRowBytes);
    if (scaledVImageBuffer == nil) {
        return nil;
    }
    
    vImage_Buffer outbuff = {scaledVImageBuffer,(NSUInteger)size.height,(NSUInteger)size.width,scaledImageRowBytes};
       
    vImage_Error scaleError = vImageScale_ARGB8888(&inbuff, &outbuff, nil, kvImageHighQualityResampling);
    if(scaleError != kvImageNoError){
        free(scaledVImageBuffer);
        scaledVImageBuffer = NULL;
        return nil;
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    CVPixelBufferRef scaledPixelBuffer = NULL;
//    CVReturn status =  CVPixelBufferCreateWithBytes(nil, (NSInteger)size.width, (NSInteger)size.height, pixelBufferType, scaledVImageBuffer, scaledImageRowBytes, releaseCallback, nil, nil, &scaledPixelBuffer);
    NSDictionary *options =@{(NSString *)kCVPixelBufferCGImageCompatibilityKey:@YES,(NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey:@YES,(NSString *)kCVPixelBufferMetalCompatibilityKey:@YES,(NSString *)kCVPixelBufferWidthKey :[NSNumber numberWithInt: size.width],(NSString *)kCVPixelBufferHeightKey: [NSNumber numberWithInt : size.height],(id)kCVPixelBufferBytesPerRowAlignmentKey:@(32)
    };
    
    CVReturn status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, size.width, size.height,pixelBufferType, scaledVImageBuffer, scaledImageRowBytes,releaseCallback,nil, (__bridge CFDictionaryRef)options, &scaledPixelBuffer);
    options = NULL;
    if (status != kCVReturnSuccess)
    {
        free(scaledVImageBuffer);
        return nil;
    }
    return scaledPixelBuffer;
}

//裁剪和缩放CVPixelBufferRef
- (CVPixelBufferRef)createCroppedPixelBufferRef:(CVPixelBufferRef)pixelBuffer cropRect:(CGRect)cropRect scaleSize:(CGSize)scaleSize context:(CIContext *)context {
//    assertCropAndScaleValid(pixelBuffer, cropRect, scaleSize);
    CIImage *image = [CIImage imageWithCVImageBuffer:pixelBuffer];
    image = [image imageByCroppingToRect:cropRect];
    
    CGFloat scaleX = scaleSize.width / CGRectGetWidth(image.extent);
    CGFloat scaleY = scaleSize.height / CGRectGetHeight(image.extent);
    
    OSType type = CVPixelBufferGetPixelFormatType(pixelBuffer);
    if (type != kCVPixelFormatType_32BGRA) {
        return nil;
    }
    
    image = [image imageByApplyingTransform:CGAffineTransformMakeScale(scaleX, scaleY)];
    
    // Due to the way [CIContext:render:toCVPixelBuffer] works, we need to translate the image so the cropped section is at the origin
    image = [image imageByApplyingTransform:CGAffineTransformMakeTranslation(-image.extent.origin.x *0.5, -image.extent.origin.y *0.5)];
    
    CVPixelBufferRef output = NULL;
    //有时候裁剪缩放过后会出现像素偏差,导致模型无法识别
//    CVPixelBufferCreate(nil,
//                        CGRectGetWidth(image.extent),
//                        CGRectGetHeight(image.extent),
//                        CVPixelBufferGetPixelFormatType(pixelBuffer),
//                        nil,
//                        &output);
    CVPixelBufferCreate(nil,
                        scaleSize.width,
                        scaleSize.height,
    CVPixelBufferGetPixelFormatType(pixelBuffer),
    nil,
    &output);
   
    if (!context) {
        context = [CIContext context];
    }
    if (output != NULL) {
        [context render:image toCVPixelBuffer:output];
    }
    
    return output;
}

//旋转CVPixelBufferRef
/* rotationConstant:
 *  0 -- rotate 0 degrees (simply copy the data from src to dest)
 *  1 -- rotate 90 degrees counterclockwise
 *  2 -- rotate 180 degress
 *  3 -- rotate 270 degrees counterclockwise
 */
- (CVPixelBufferRef)rotateBuffer:(CVPixelBufferRef)imageBuffer withConstant:(uint8_t)rotationConstant
{
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
//    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    OSType pixelFormatType              = CVPixelBufferGetPixelFormatType(imageBuffer);
    NSAssert(pixelFormatType == kCVPixelFormatType_32BGRA, @"Code works only with 32ARGB format. Test/adapt for other formats!");

    const size_t kAlignment_32ARGB      = 32;
    const size_t kBytesPerPixel_32ARGB  = 4;

    size_t bytesPerRow                  = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width                        = CVPixelBufferGetWidth(imageBuffer);
    size_t height                       = CVPixelBufferGetHeight(imageBuffer);

    BOOL rotatePerpendicular            = (rotationConstant == 1) || (rotationConstant == 3); // Use enumeration values here
    const size_t outWidth               = rotatePerpendicular ? height : width;
    const size_t outHeight              = rotatePerpendicular ? width  : height;

    size_t bytesPerRowOut               = kBytesPerPixel_32ARGB * ceil(outWidth * 1.0 / kAlignment_32ARGB) * kAlignment_32ARGB;

    const size_t dstSize                = bytesPerRowOut * outHeight * sizeof(unsigned char);

    unsigned char *srcBuff              = CVPixelBufferGetBaseAddress(imageBuffer);
    
    unsigned char *dstBuff              = malloc(dstSize);
    
    vImage_Buffer inbuff                = {srcBuff, height, width, bytesPerRow};
    vImage_Buffer outbuff               = {dstBuff, outHeight, outWidth, bytesPerRowOut};

    uint8_t bgColor[4]                  = {0, 0, 0, 0};

    vImage_Error err                    = vImageRotate90_ARGB8888(&inbuff, &outbuff, rotationConstant, bgColor, kvImageNoFlags);
    if (err != kvImageNoError)
    {
        free(dstBuff);
        return nil;
    }
   
    CVPixelBufferRef rotatedBuffer      = NULL;
    CVPixelBufferCreateWithBytes(NULL,
                                 outWidth,
                                 outHeight,
                                 pixelFormatType,
                                 dstBuff,
                                 bytesPerRowOut,
                                 releaseCallback,
                                 NULL,
                                 NULL,
                                 &rotatedBuffer);
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
//    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    return rotatedBuffer;
}

static inline size_t vImageByteAlign(size_t size, size_t alignment) {
    return ((size + (alignment - 1)) / alignment) * alignment;
}
void releaseCallback(void *releaseRefCon, const void *baseAddress) {
        free((void *)baseAddress);
}

#pragma mark----TensorFlowLite
- (void)setupInterpreter
{
    NSError *error;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"detect" ofType:@"tflite"];
    //初始化识别器,需要传入训练模型的路径,还可以传options
    self.interpreter = [[TFLInterpreter alloc] initWithModelPath:path error:&error];
    
    if (![self.interpreter allocateTensorsWithError:&error]) {
        NSLog(@"Create interpreter error: %@", error);
    }
}

#pragma mark------Camera
- (void)setupCamera
{
    self.session = [[AVCaptureSession alloc] init];
    [self.session setSessionPreset:AVCaptureSessionPresetHigh];//开启高质量模式,一般使用16:9
//    [self.session setSessionPreset:AVCaptureSessionPreset640x480];//如果需要4:3最好设置,避免自己裁切
    
    self.inputDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];//默认
//    self.inputDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];//指定广角模式和镜头

    NSError *error;
    self.deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:self.inputDevice error:&error];
    
    if ([self.session canAddInput:self.deviceInput]) {
        [self.session addInput:self.deviceInput];
    }
    
    self.previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.session];
    [self.previewLayer setVideoGravity:AVLayerVideoGravityResizeAspectFill];
//    [self.previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];按比例拉伸
    CALayer *rootLayer = [[self view] layer];
    [rootLayer setMasksToBounds:YES];
    CGRect frame = self.view.frame;
    [self.previewLayer setFrame:frame];
//    [self.previewLayer setFrame:CGRectMake(0, 0, frame.size.width, frame.size.width * 4 / 3)];
    [rootLayer insertSublayer:self.previewLayer atIndex:0];
    //添加绘制图层
    self.overlayView = [[OverlayView alloc] initWithFrame:self.previewLayer.bounds];
    [self.view addSubview:self.overlayView];
    self.overlayView.clearsContextBeforeDrawing = YES;//设置清空画布上下文
    
    AVCaptureVideoDataOutput *videoDataOutput = [AVCaptureVideoDataOutput new];
    
    NSDictionary *rgbOutputSettings = [NSDictionary
                                       dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                       forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    [videoDataOutput setVideoSettings:rgbOutputSettings];
    [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
    dispatch_queue_t videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
    [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    if ([self.session canAddOutput:videoDataOutput])
        [self.session addOutput:videoDataOutput];
//    [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];
    [videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoOrientation =  AVCaptureVideoOrientationPortrait;//指定镜头方向
    
    [self.session startRunning];
 
}


#pragma mark-----save Image to check
- (void)image:(UIImage *)image didFinishSavingWithError:(NSError *)error contextInfo:(void *)contextInfo
{
    NSLog(@"image = %@, error = %@, contextInfo = %@", image, error, contextInfo);
}

- (UIImage *)imageFromSampleBuffer:(CVPixelBufferRef) imageBuffer {
    // 为媒体数据设置一个CMSampleBuffer的Core Video图像缓存对象
//    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // 锁定pixel buffer的基地址
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // 得到pixel buffer的基地址
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // 得到pixel buffer的行字节数
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    // 得到pixel buffer的宽和高
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // 创建一个依赖于设备的RGB颜色空间
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // 用抽样缓存的数据创建一个位图格式的图形上下文（graphics context）对象
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    // 根据这个位图context中的像素数据创建一个Quartz image对象
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // 解锁pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // 释放context和颜色空间
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // 用Quartz image创建一个UIImage对象image
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    
    // 释放Quartz image对象
    CGImageRelease(quartzImage);
    
    return (image);
}


#pragma mark---lazy init
-(HFCameraHelper *)helper
{
    if (!_helper) {
        _helper = [HFCameraHelper new];
    }
    return _helper;
}

- (CIContext *)context
{
    if (!_context) {
        _context = [CIContext context];
    }
    return _context;
}



@end
