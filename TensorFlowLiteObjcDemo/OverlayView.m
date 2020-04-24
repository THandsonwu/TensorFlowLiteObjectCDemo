//
//  OverlayView.m
//  TensorFlowLiteObjcDemo
//
//  Created by tanzhiwu on 2020/4/24.
//  Copyright © 2020 tanzhiwu. All rights reserved.
//

#import "OverlayView.h"
@implementation Overlayer

@end
@interface OverlayView ()
@property (nonatomic, assign) CGFloat cornerRadius;
@property (nonatomic, assign) CGFloat stringBgalpha;
@property (nonatomic, assign) CGFloat lineWidth;
@property (nonatomic, strong) UIColor *stringFontColor;
@property (nonatomic, assign) CGFloat stringHorizontalSpacing;
@property (nonatomic, assign) CGFloat stringVerticalSpacing;
@end
@implementation OverlayView

- (instancetype)initWithFrame:(CGRect)frame
{
    if (self = [super initWithFrame:frame]) {
        self.cornerRadius = 10.0;
        self.stringBgalpha = 0.7;
        self.lineWidth = 3;
        self.stringFontColor = UIColor.whiteColor;
        self.stringHorizontalSpacing = 13.0;
        self.stringVerticalSpacing = 7.0;
        self.backgroundColor = [UIColor clearColor];
    }
    return self;
}
// Only override drawRect: if you perform custom drawing.
// An empty implementation adversely affects performance during animation.
- (void)drawRect:(CGRect)rect {
    // Drawing code
    
    for (Overlayer *lay in self.overlays) {
        [self drawBorders:lay];
        [self drawbackground:lay];
        [self drawName:lay];
    }
}

- (void)drawBorders:(Overlayer *)overlay
{
    UIBezierPath *path = [UIBezierPath bezierPathWithRect:overlay.borderRect];
    path.lineWidth = self.lineWidth;
    [overlay.color setStroke];
    [path stroke];
}

- (void)drawbackground:(Overlayer *)overlay
{
    
    CGRect stringBgRect = CGRectMake(overlay.borderRect.origin.x, overlay.borderRect.origin.y, 2 * self.stringHorizontalSpacing + overlay.nameStringSize.width, 2 * self.stringVerticalSpacing + overlay.nameStringSize.height);
    
    UIBezierPath *path = [UIBezierPath bezierPathWithRect:stringBgRect];
    [[overlay.color colorWithAlphaComponent:self.stringBgalpha] setFill];
    [path fill];
}

- (void)drawName:(Overlayer *)overlay
{
    //增加文字旋转
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSaveGState(context);
    CGRect stringRect = CGRectMake(overlay.borderRect.origin.x + self.stringHorizontalSpacing, overlay.borderRect.origin.y + self.stringVerticalSpacing, overlay.nameStringSize.width, overlay.nameStringSize.height);
    //旋转绘制暂时没弄
    NSAttributedString *attributedString = [[NSAttributedString alloc] initWithString:overlay.name attributes:@{NSForegroundColorAttributeName: self.stringFontColor,NSFontAttributeName:overlay.font}];
    [attributedString drawInRect:stringRect];
    CGContextRestoreGState(context);
}

@end
