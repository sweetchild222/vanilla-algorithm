package org.example;

import org.jetbrains.annotations.NotNull;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Map;

public class ImageResize {

    public static @NotNull BufferedImage scaleToFit(BufferedImage srcImage, int targetWidth, int targetHeight){

        final float widthRatio = (float) targetWidth / srcImage.getWidth();
        final float heightRatio = (float) targetHeight / srcImage.getHeight();

        int scaledWidth = widthRatio >= heightRatio ? targetWidth : Math.round(srcImage.getWidth() * heightRatio);
        int scaledHeight = widthRatio <= heightRatio ? targetHeight : Math.round(srcImage.getHeight() * widthRatio);

        final Image scaledImage = srcImage.getScaledInstance(scaledWidth, scaledHeight, Image.SCALE_SMOOTH);
        final BufferedImage newImage = new BufferedImage(scaledWidth, scaledHeight, BufferedImage.TYPE_INT_ARGB);

        final Graphics2D g2d = newImage.createGraphics();
        g2d.drawImage(scaledImage, 0, 0, null);
        g2d.dispose();

        return newImage;
    }


    public static @NotNull BufferedImage crop(BufferedImage srcImage, int targetWidth, int targetHeight) {

        final int srcWidth = srcImage.getWidth();
        final int srcHeight = srcImage.getHeight();

        final int cropX = Math.max(0, (srcWidth - targetWidth) / 2);
        final int cropY = Math.max(0, (srcHeight - targetHeight) / 2);

        final int finalWidth = Math.min(targetWidth, srcWidth - cropX);
        final int finalHeight = Math.min(targetHeight, srcHeight - cropY);

        return srcImage.getSubimage(cropX, cropY, finalWidth, finalHeight);
    }


    public static BufferedImage [] resize(BufferedImage srcImage, int orientation, int[][] sizeList) {

        final BufferedImage rotatedSrcImage = rotate(srcImage, orientation);

        if (rotatedSrcImage == null)
            return null;

        final ArrayList<BufferedImage> list = new ArrayList<>();

        for (int[] size : sizeList) {

            final int targetWidth = size[0];
            final int targetHeight = size[1];

            if (rotatedSrcImage.getWidth() == targetWidth && rotatedSrcImage.getHeight() == targetHeight) {
                list.add(rotatedSrcImage);
            } else {

                final BufferedImage scaledImage = scaleToFit(rotatedSrcImage, targetWidth, targetHeight);

                BufferedImage croppedImage = crop(scaledImage, targetWidth, targetHeight);

                list.add(croppedImage);
            }
        }

        return list.toArray(BufferedImage[]::new);
    }


    private static BufferedImage rotate(BufferedImage srcImage, int orientation) {

        final BufferedImage newImage;

        if (orientation == 5 || orientation == 6 || orientation == 7 || orientation == 8)
            newImage = new BufferedImage(srcImage.getHeight(), srcImage.getWidth(), srcImage.getType());
        else if (orientation == 2 || orientation == 3 || orientation == 4)
            newImage = new BufferedImage(srcImage.getWidth(), srcImage.getHeight(), srcImage.getType());
        else    // orientation == 1
            return srcImage;

        final Graphics2D graphics = (Graphics2D) newImage.getGraphics();

        if(orientation == 3 || orientation == 5 || orientation == 6 || orientation == 7 || orientation == 8) {

            final int radians = (Map.of(3, 180, 5, 270, 6, 90, 7, 270, 8, 270)).getOrDefault(orientation, 0);

            graphics.rotate(Math.toRadians(radians), (double)newImage.getWidth() / 2, (double)newImage.getHeight() / 2);
            graphics.translate((newImage.getWidth() - srcImage.getWidth()) / 2, (newImage.getHeight() - srcImage.getHeight()) / 2);
        }


        if(orientation == 3 || orientation == 6 || orientation == 8)
            graphics.drawRenderedImage(srcImage, null);
        else if(orientation == 2 || orientation == 5)
            graphics.drawImage(srcImage, srcImage.getWidth(), 0, 0, srcImage.getHeight(), 0, 0, srcImage.getWidth(), srcImage.getHeight(), null);
        else //if(orientation == 4 || orientation == 7)
            graphics.drawImage(srcImage, 0, srcImage.getHeight(), srcImage.getWidth(), 0, 0, 0, srcImage.getWidth(), srcImage.getHeight(), null);

        return newImage;
    }
}
