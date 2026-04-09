package org.example;


import org.example.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;


public class Main {

    static void main() {

        final int[][] sizeList = {{256, 256}, {128, 128}, {64, 64}, {32, 32}};
        final String filePath = "D:/images";
        final ImageSetWriter writer = new ImageSetWriter(filePath);

        try {

            URL url = new URL("https://upload.wikimedia.org/wikipedia/commons/6/60/%22_Le_Sanitor_%22%2C_d%C3%A9sinfectant_sans_odeur%2C_sant%C3%A9_par_l%27hygi%C3%A8ne_%28...%29%2C_antiseptique%2C_antiputride%2C_anti%C3%A9pid%C3%A9mique._Audibert_%26_Cie%2C_usine_et_administration_68%2C_Boul%28var%29d_St_Marcel%2C_4%2C_rue_Scipion%2C_Paris._En_vente..._-_btv1b9005629d.jpg");

            HttpURLConnection connection = (HttpURLConnection) url.openConnection();

            connection.setRequestProperty("User-Agent", "Mozilla/5.0");

            BufferedImage image = ImageIO.read(connection.getInputStream());

            final String id = writeImageSet(image, sizeList, writer);

        } catch (IOException e) {

            e.printStackTrace();
        }
    }

    private static String writeImageSet(BufferedImage srcImage, int [][] supportSizeList, ImageSetWriter setWriter){

        final BufferedImage [] imageSet = ImageResize.resize(srcImage, 1, supportSizeList);

        if(imageSet == null)
            return null;

        if(imageSet.length != supportSizeList.length)
            return null;

        return setWriter.write(imageSet);
    }
}
