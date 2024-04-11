def CE(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img_out = img
    
    # Get saturation and brightness in HSL space
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0
    
    # s = L<0.5 ? s1 : s2
    mask_1 = L < 0.5
    s1 = delta/(value)
    s2 = delta/(2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    
    # The increment is greater than 0, and the saturation index is enhanced.
    if increment >= 0 :
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        
        alpha = 1/alpha -1 
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        
    # Increment less than 0, saturation linearly decays
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    
    img_out = img_out/255.0
    
    # RGB color upper and lower limit processing (less than 0, take 0, greater than 1, take 1)
    mask_3 = img_out  < 0 
    mask_4 = img_out  > 1
    img_out = img_out * (1-mask_3)
    img_out = img_out * (1-mask_4) + mask_4
    
    return img_out
