//
//  HFCameraHelper.h
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface HFCameraHelper : NSObject
- (void)startUpdateAccelerometerResult:(void (^)(UIImageOrientation orientation))result;
@end

NS_ASSUME_NONNULL_END
