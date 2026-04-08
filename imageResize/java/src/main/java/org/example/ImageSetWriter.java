package org.example;


import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

public class ImageSetWriter {

    private final String path;

    public ImageSetWriter(String path){

        this.path = path;
    }


    public String write(BufferedImage[] imageSet) {

        String path = createFolder();

        if(path == null)
            return null;

        final String id = generateID();

        for(BufferedImage image : imageSet) {

            final String fileName = id + "_" + image.getWidth() + "x" + image.getHeight() + ".png";

            final String filePath = path + "/" + fileName;

            try {

                boolean success = ImageIO.write(image, "PNG", new File(filePath));

                if(!success)
                    return null;
            }
            catch (IOException e){
                return null;
            }
        }

        return id + ".png";
    }


    private String createFolder(){

        Path directoryPath = Paths.get(this.path);

        try {

            if(!Files.exists(directoryPath))
                Files.createDirectories(directoryPath);

            return directoryPath.toAbsolutePath().toString();

        }catch(IOException e) {
            return null;
        }
    }


    private String generateID(){

        UUID uuid = UUID.randomUUID();

        LocalDateTime now = LocalDateTime.now();

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");

        return now.format(formatter) + "-" + uuid;
    }
}
