//
//  HFInference.h
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface HFInference : NSObject
@property (nonatomic, assign) CGFloat confidence;
@property (nonatomic, strong) NSString *className;
@property (nonatomic, assign) NSInteger index;
@property (nonatomic, assign) CGRect boundingRect;
@property (nonatomic, strong) UIColor *displayColor;
@end

NS_ASSUME_NONNULL_END
