//
//  OverlayView.h
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import <UIKit/UIKit.h>
/*
AVCaptureVideoOrientationPortrait           = 1,
AVCaptureVideoOrientationPortraitUpsideDown = 2,
AVCaptureVideoOrientationLandscapeRight     = 3,
AVCaptureVideoOrientationLandscapeLeft      = 4,
*/
NS_ASSUME_NONNULL_BEGIN

@interface Overlayer : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) CGRect borderRect;
@property (nonatomic, assign) CGSize nameStringSize;
@property (nonatomic, strong) UIColor *color;
@property (nonatomic, strong) UIFont *font;
@property (nonatomic, assign) NSUInteger nameDirection;
@end

@interface OverlayView : UIView
@property (nonatomic, strong) NSMutableArray<Overlayer *>* overlays;
@end

NS_ASSUME_NONNULL_END
