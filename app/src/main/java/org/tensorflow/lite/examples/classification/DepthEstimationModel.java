package org.tensorflow.lite.examples.classification;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

public class DepthEstimationModel {
    protected Interpreter tflite;

    protected DepthEstimationModel(Activity activity) throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(activity, getModelPath());

        Interpreter.Options tfliteOptions = new Interpreter.Options();
        GpuDelegate gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        tfliteOptions.setNumThreads(4);

        tflite = new Interpreter(modelBuffer, tfliteOptions);
    }

    protected String getModelPath() {
        return "pydnet++.tflite";
    }

    public void close() {
        tflite.close();
        tflite = null;
    }

    public float[] inference(ByteBuffer input, int height, int width) {
        float[][][][] tmpOutput = new float[1][height][width][1];
        tflite.run(input, tmpOutput);

        float[] output = new float[width * height];
        for (int row=0; row < height; row++){
            for (int col=0; col < width; col++) {
                output[row * width + col] = tmpOutput[0][row][col][0];
            }
        }
        return output;
    }

    public static float[][][][] getPixelFromBitmap(Bitmap frame){
        int[] pixels = new int[frame.getWidth() * frame.getHeight()];
        frame.getPixels(pixels, 0, frame.getWidth(), 0, 0, frame.getWidth(), frame.getHeight());

        float[][][][] output = new float[1][frame.getHeight()][frame.getWidth()][3];
        int y = 0, x = 0, c = 0;
        for (int pixel : pixels) {
            output[0][y][x][0] = Color.red(pixel) / (float)255.;
            output[0][y][x][1] = Color.green(pixel) / (float)255.;
            output[0][y][x][2] = Color.blue(pixel) / (float)255.;
            x += 1;
            c += 1;
            if (x == frame.getWidth()) {
                y += 1;
                x = 0;
            }
        }
        return output;
    }

    public void inferenceBitmap(Bitmap inputBitmap, Bitmap outputBitmap) {
        int width = 640, height = 448;
        Bitmap resizedBitmap = Bitmap.createBitmap(inputBitmap, 0, 0, width, height);

        Tensor inputTensor = tflite.getInputTensor(tflite.getInputIndex("im0"));
        Tensor outputTensor = tflite.getOutputTensor(tflite.getOutputIndex("PSD/resize/ResizeBilinear"));

        TensorImage inputImageBuffer = new TensorImage(inputTensor.dataType());
        inputImageBuffer.load(resizedBitmap);
        ByteBuffer inputBuffer = inputImageBuffer.getBuffer();
//        float[][][][] input = getPixelFromBitmap(inputBitmap);

        float[] output = inference(inputBuffer, height, width);
        int[] colors = applyColorMap(output);
        outputBitmap.setPixels(colors, 0, width, 0, 0, width, height);
    }

    private int[] applyColorMap(float[] output) {
        int[] color = new int[output.length];
        for (int i=0; i < output.length; i++) {
            float x = (float) (output[i] / 1024.0 * 256);
            int red = (int) x;
            int green = red, blue = red;
//            int red = (int) (-0.0045 * Math.pow(x, 2) + 0.2337 * x + 248.25);
//            int green = (int) (2e-5 * Math.pow(x, 3) - 0.0023 * Math.pow(x, 2) + 1.3374 * x + 243.68);
//            int blue = (int) (-4e-5 * Math.pow(x, 3) + 0.0112 * Math.pow(x, 2) - 0.1947 * x + 37.514);
//            red = Math.max(Math.min(red, 255), 0);
//            green = Math.max(Math.min(green, 255), 0);
//            blue = Math.max(Math.min(blue, 255), 0);
            color[i] = 0xff << 24 | (red & 0xff) << 16 | (green & 0xff) << 8 | (blue & 0xff);
        }
        return color;
    }
}
