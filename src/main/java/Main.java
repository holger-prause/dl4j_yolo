import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
//import org.nd4j.jita.conf.CudaEnvironment;

import java.io.File;
import java.util.List;

/**
 * Created by Holger on 20.05.2018.
 */
public class Main {
    public static void main(String[] args) {
        YOLOModel yoloModel = new YOLOModel();
        List<DetectedObject> detectedObjects = yoloModel.detect(new File("car.png"), 0.8);

    }
}
