//
//  HFCameraHelper.m
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import "HFCameraHelper.h"
#import <CoreMotion/CoreMotion.h>
#import <AVFoundation/AVFoundation.h>
@interface HFCameraHelper()
@property (nonatomic,strong) CMMotionManager *motionManager;
@property (nonatomic,assign) UIImageOrientation orientation;
@end

@implementation HFCameraHelper
- (CMMotionManager *)motionManager
{
    if (!_motionManager)
    {
        _motionManager = [[CMMotionManager alloc]init];
    }
    return _motionManager;
}

//启动重力加速计
- (void)startUpdateAccelerometerResult:(void (^)(UIImageOrientation orientation))result
{
    if ([self.motionManager isAccelerometerAvailable]){
        [self.motionManager setAccelerometerUpdateInterval:0.1];
        [self.motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue mainQueue] withHandler:^(CMAccelerometerData *accelerometerData, NSError *error)
         {
             double x = accelerometerData.acceleration.x;
             double y = accelerometerData.acceleration.y;
             if (fabs(y) >= fabs(x))
             {
                 if (y >= 0){
                     //Down
                     if (result) {
                         result(UIImageOrientationDown);
                     }
                 }
                 else{
                     //Portrait
                     if (result) {
                         result(UIImageOrientationUp);
                     }
                 }
             }
             else
             {
                 if (x >= 0){
                     //Right
                     if (result) {
                         result(UIImageOrientationRight);
                     }
                 }
                 else{
                     //Left
                     if (result) {
                         result(UIImageOrientationLeft);
                     }
                 }
             }
         }];
    }
}
@end
