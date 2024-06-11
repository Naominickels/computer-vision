package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Size;
import android.view.Surface;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.app.Fragment;
import android.widget.ToggleButton;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Semaphore;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {

    public static class Models {
        boolean loaded = false;
        int curModel = 0;
        ArrayList<Module> modules_classify = new ArrayList<>();
        ArrayList<Module> modules_classify_dropout = new ArrayList<>();
        ArrayList<Integer> modules_size = new ArrayList<>();
        ArrayList<float[]> modules_mean = new ArrayList<>();
        ArrayList<float[]> modules_std = new ArrayList<>();
        ArrayList<String> modules_name = new ArrayList<>();
        public float[][] mean_feature_map_0;
        public float[][] mean_feature_maps;
        public float[][] covar_0_inverse;
        public float[][] covar_inverse;

        int oodInputHeight;
        int oodInputWidth;
        public float[] ood_mean;
        public float[] ood_std;
        Module module_feature;
        public void addModule(Module classify, Module dropout, int imageSize, float[] mean, float[] std, String moduleName) {
            modules_classify.add(classify);
            modules_classify_dropout.add(dropout);
            modules_size.add(imageSize);
            modules_mean.add(mean);
            modules_std.add(std);
            modules_name.add(moduleName);
        }

        public void nextModule() {
            curModel = (curModel + 1) % modules_classify.size();
        }

        public Module getCurModule(boolean dropout) {
            return  dropout ? modules_classify_dropout.get(curModel) : modules_classify.get(curModel);
        }

        public int getCurSize() {
            return modules_size.get(curModel);
        }

        public float[] getCurMean() {return modules_mean.get(curModel);}

        public float[] getCurStd() {return modules_std.get(curModel);}

        public String getCurName() {
            return modules_name.get(curModel);
        }
    }

    public final Models models = new Models();

    public static final int IMAGE_CAPTURE_CODE = 654;

    FrameLayout imageFrameLayout;
    ImageView imageView;
    Button loadImage;
    Button detect;
    TextView oodText;
    TextView resultText;
    TextView resultText2;
    ToggleButton dropout;
    ToggleButton realtime;
    Button switchModel;

    Uri imageUri;
    Bitmap curImage = null;
    int rectHeight;
    int rectWidth;
    float resizeRatio;

    final float[] normalMean = {0.5f, 0.5f, 0.5f};
    final float[] normalStd = {0.5f, 0.5f, 0.5f};
    final float[] imageNetMean = {0.485f, 0.456f, 0.406f};
    final float[] imageNetStd = {0.229f, 0.224f, 0.225f};
    final float[] identityMean = {0.f, 0.f, 0.f};
    final float[] identityStd = {1.f, 1.f, 1.f};

    final String[] oneHotDecode = {
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise"};

    int previewHeight;
    int previewWidth;
    int[] rgbBytes;
    int sensorOrientation;

    boolean imageCaptured = false;

    final Semaphore readyToInfer = new Semaphore(1);

    private void openCamera() {
        CameraManager manager = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[0];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Fragment fragment;
        CameraConnectionFragment camera2Fragment = CameraConnectionFragment.Companion.newInstance(
                (CameraConnectionFragment.ConnectionCallback) (size, cameraRotation) -> {
                    previewHeight = size.getHeight();
                    previewWidth = size.getWidth();
                    sensorOrientation = cameraRotation - getScreenOrientation();
                },
                this,
                R.layout.fragment_camera_connection,
                new Size(600, 600)
        );
        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.frame, fragment).commit();
    }

    protected Integer getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_0:
                return 0;
        }
        return 0;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == IMAGE_CAPTURE_CODE && grantResults.length > 0){
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                resultText.setText("");
                openCamera();
            } else {
                resultText.setText("PLEASE GRANT CAMERA PERMISSIONS BY TOUCHING AREA ABOVE");
            }
        }
    }

    public String assetFilePath(String assetName) {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                try {
                    Files.write(file.toPath(), new byte[0], StandardOpenOption.TRUNCATE_EXISTING);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            //return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

    private MappedByteBuffer loadModelFile(String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor=this.getAssets().openFd(modelName);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }

    private void writeBitmap(Bitmap bitmap, String name) {
        String path = Environment.getExternalStorageDirectory().toString() + "/Download";
        OutputStream fOut = null;
        Integer counter = 0;
        File file = new File(path, name + ".png"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
        try {
            fOut = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fOut);
            fOut.flush();
            fOut.close();
            MediaStore.Images.Media.insertImage(getContentResolver(),file.getAbsolutePath(),file.getName(),file.getName());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

//    private TensorImage deNormalize(TensorImage tensorImage, float[] mean, float[] std) {
//        float[] invMean = new float[mean.length];
//        float[] invStd = new float[mean.length];
//
//        for (int i = 0; i < mean.length; i++) {
//            invMean[i] = - mean[i] / std[i];
//            invStd[i] = 1 / std[i];
//        }
//        NormalizeOp deNorm = new NormalizeOp(invMean, invStd);
//        TensorBuffer tensorBuffer = deNorm.apply(tensorImage.getTensorBuffer());
//        tensorImage.load(tensorBuffer);
//        return tensorImage;
//    }
//
//    private Bitmap deNormalize(Bitmap bitmap, float[] mean, float[] std) {
//        return deNormalize(TensorImage.fromBitmap(bitmap), mean, std).getBitmap();
//    }

    private void requestAppPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                String[] permission = {Manifest.permission.CAMERA};
                requestPermissions(permission, IMAGE_CAPTURE_CODE);
            }
        }
    }

    private void drawRectangle(Canvas canvas, int h, int w) {
        int startX = (w / 2 - rectWidth / 2);
        int startY = (h / 2 - rectHeight / 2);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.BLUE);
        paint.setStrokeWidth(10);
        canvas.drawRect(startX, startY, startX + rectWidth, startY + rectHeight, paint);
    }

//    private void setTensorImageAsBackground(TensorImage tensorImage) {
//        Bitmap tempBitmap = Bitmap.createBitmap(curImage.getWidth(), curImage.getHeight(), Bitmap.Config.RGB_565);
//        Canvas canvas = new Canvas(tempBitmap);
//        Rect dst = new Rect();
//        dst.set((curImage.getWidth() - inputWidth) / 2, (curImage.getHeight() - inputHeight) / 2,
//                (curImage.getWidth() + inputWidth) / 2, (curImage.getHeight() + inputHeight) / 2);
//        canvas.drawBitmap(deNormalize(tensorImage, normalizeMeanTf, normalizeStdTf).getBitmap(), null, dst, null);
//        runOnUiThread(() -> {
//            imageView.setBackground(new BitmapDrawable(getResources(), tempBitmap));
//        });
//        //writeBitmap(deNormalize(tensorImage, normalizeMeanTf, normalizeStdTf).getBitmap(), "test_out");
//    }

    public Bitmap getCurImageBitmap(int width, int height) {
        int w = curImage.getWidth();
        int h = curImage.getHeight();
        Bitmap resized = Bitmap.createScaledBitmap(curImage, width, height, true);

        return resized;
    }

    public Bitmap toGrayscale(Bitmap input) {
        Bitmap bmpGrayscale = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(input, 0, 0, paint);
        return bmpGrayscale;
    }

//    public TensorImage getCurImageTensorImage() {
//        Bitmap bitmap = getCurImageBitmap(segmentInputWidth, segmentInputHeight);
//        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
//        tensorImage.load(bitmap);
//        NormalizeOp normalizeOp = new NormalizeOp(normalizeMeanTf, normalizeStdTf);
//        TensorBuffer tensorBuffer = normalizeOp.apply(tensorImage.getTensorBuffer());
//        tensorImage.load(tensorBuffer);
//        return tensorImage;
//    }

    public float[][] readMatrix(String path) {
        try (BufferedReader fileReader = new BufferedReader(new FileReader(path))) {
            int n = Integer.parseInt(fileReader.readLine());
            int m = Integer.parseInt(fileReader.readLine());
            float[][] data = new float[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++)
                    data[i][j] = Float.parseFloat(fileReader.readLine());
            }
            return data;
        } catch (IOException e) {
            return null;
        }
    }

    public float[][] matrixMul(float[][] m1, float[][] m2) {
        if (m1[0].length != m2.length)
            throw new RuntimeException();

        float[][] res = new float[m1.length][m2[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m2[0].length; j++)
                res[i][j] = 0.f;
        }

        for (int k = 0; k < m1[0].length; k++) {
            for (int i = 0; i < m1.length; i++) {
                for (int j = 0; j < m2[0].length; j++)
                    res[i][j] += m1[i][k] * m2[k][j];
            }
        }
        return res;
    }

    public float[][] matrixDif(float[][] m1, float[][] m2) {
        if (m1.length != m2.length || m1[0].length != m2[0].length)
            throw new RuntimeException();

        float[][] res = new float[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++)
                res[i][j] = m1[i][j] - m2[i][j];
        }
        return res;
    }

    public float[][] matrixTranspose(float[][] m) {
        float[][] res = new float[m[0].length][m.length];

        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++)
                res[j][i] = m[i][j];
        }
        return res;
    }
    public float mahalanobis_distance(float[][] feature_map, float[][] mean_feature_map, float[][] covar_inverse) {
        float[][] adjusted_feature_map = matrixDif(feature_map, mean_feature_map);
        return matrixMul(matrixMul(adjusted_feature_map, covar_inverse), matrixTranspose(adjusted_feature_map))[0][0];
    }

    public float[][] vecToMatrix(float[] v) {
        float[][] res = new float[1][v.length];
        System.arraycopy(v, 0, res[0], 0, v.length);
        return res;
    }

    public float rmd_confidence(float[][] feature_map, float[][] mean_feature_maps, float[][] mean_feature_map_0, float[][] covar_inverse, float[][] covar_0_inverse) {
        float md_0 = mahalanobis_distance(feature_map, mean_feature_map_0, covar_0_inverse);

        ArrayList<Float> rmd = new ArrayList<>();
        for (float[] mean_feature_map : mean_feature_maps)
            rmd.add(mahalanobis_distance(feature_map, vecToMatrix(mean_feature_map), covar_inverse) - md_0);

        float min_val = rmd.get(0);
        for (int i = 1; i < rmd.size(); i++)
            min_val = rmd.get(i) < min_val ? rmd.get(i) : min_val;
        return -min_val;
    }

    public float[] softmax(float[] v) {
        float[] exps = new float[v.length];
        float sum = 0.f;

        for (int i = 0; i < v.length; i++) {
            exps[i] = (float) Math.exp((double) v[i]);
            sum += exps[i];
        }

        for (int i = 0; i < exps.length; i++)
            exps[i] /= sum;

        return exps;
    }

    public void runOOD(boolean infoText) {
        if (infoText) {
            runOnUiThread(() -> {
                resultText.setText("Analyzing...");
            });
        }
        // compute Relative Mahalanobis Distance
//                    Tensor feature = models.module_feature.forward(IValue.from(inputTensor)).toTensor();
        Bitmap inputImage = getCurImageBitmap(models.oodInputWidth, models.oodInputHeight);
        Bitmap grayscaleImage = toGrayscale(inputImage);

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(grayscaleImage, models.ood_mean, models.ood_std);

        Tensor feature = models.module_feature.forward(IValue.from(inputTensor)).toTensor();
        float confidence = rmd_confidence(vecToMatrix(feature.getDataAsFloatArray()),
                models.mean_feature_maps,
                models.mean_feature_map_0,
                models.covar_inverse,
                models.covar_0_inverse);

        System.out.println("IND confidence score: " + String.format("%.02f", confidence));
        float ood_threshold = -2000.f;
        runOnUiThread(() -> {
            if (infoText)
                resultText.setText("");
            if (confidence > ood_threshold)
                oodText.setText("No face in frame...");
            else
                oodText.setText("Face detected");
        });
    }
    public void runInference()
    {
        int nr_passes = dropout.isChecked() ? 10 : 1;
        float[] result = new float[7];
        Arrays.fill(result, 0.f);

        int curSize = models.getCurSize();
        Bitmap inputImage = getCurImageBitmap(curSize, curSize);
        Bitmap grayscaleImage = toGrayscale(inputImage);

//        writeBitmap(grayscaleImage, "test.png");

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(grayscaleImage, models.getCurMean(), models.getCurStd());

        for (int i = 0; i < nr_passes; i++) {
            float[] out = models.getCurModule(dropout.isChecked()).forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
            for (int j = 0; j < out.length; j++)
                result[j] += out[j];
        }
        for (int j = 0; j < result.length; j++)
            result[j] /= nr_passes;

        result = softmax(result);

        System.out.println("Result vector is: " + Arrays.toString(result));
        int maxPos = 0, secondMaxPos = 0;
        float maxVal = 0, secondMaxVal = 0;
        for (int i = 0; i < result.length; i++) {
            if (result[i] > maxVal) {
                secondMaxVal = maxVal;
                secondMaxPos = maxPos;
                maxVal = result[i];
                maxPos = i;
            } else if (result[i] > secondMaxVal) {
                secondMaxVal = result[i];
                secondMaxPos = i;
            }
        }
        System.out.println("Max pos is " + maxPos + " and second max pos is " + secondMaxPos);
        int finalMaxPos = maxPos;
        int finalSecondMaxPos = secondMaxPos;
        int maxPercentage = (int)(maxVal * 100);
        int secondMaxPercentage = (int)(secondMaxVal * 100);
        runOnUiThread(() -> {
            resultText.setText(oneHotDecode[finalMaxPos] + " " + maxPercentage + "%");
            if (secondMaxPercentage != 0)
                resultText2.setText(oneHotDecode[finalSecondMaxPos] + " " + secondMaxPercentage + "%");
            else
                resultText2.setText("");
        });
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageFrameLayout = (FrameLayout)findViewById(R.id.frame);
        imageView = (ImageView)findViewById(R.id.image);
        loadImage = (Button)findViewById(R.id.button);
        detect = (Button)findViewById(R.id.detect);

        oodText = (TextView)findViewById(R.id.ood_text);
        resultText = (TextView)findViewById(R.id.result_text);
        resultText2 = (TextView)findViewById(R.id.result_text2);
        dropout = (ToggleButton)findViewById(R.id.dropout);
        realtime = (ToggleButton)findViewById(R.id.realtime) ;
        switchModel = (Button)findViewById(R.id.switch_model);

        AsyncTask.execute(() -> {
            runOnUiThread(() -> {
                resultText.setText("Loading models...");
            });
            models.module_feature = Module.load(assetFilePath("mobilenet_v2_0.75_160_feats_mobile.pt"));
            models.ood_mean = normalMean;
            models.ood_std = normalStd;
            models.oodInputHeight = 160;
            models.oodInputWidth = 160;
            models.mean_feature_map_0 = readMatrix(assetFilePath("mean_feature_map_0.matrix"));
            models.mean_feature_maps = readMatrix(assetFilePath("mean_feature_maps.matrix"));
            models.covar_0_inverse = readMatrix(assetFilePath("covar_0_inverse.matrix"));
            models.covar_inverse = readMatrix(assetFilePath("covar_inverse.matrix"));

            Module classify, dropout;

            classify = Module.load(assetFilePath("mobilenet_v2_0.75_160_full_mobile.pt"));
            dropout = Module.load(assetFilePath("mobilenet_v2_0.75_160_dropout_mobile.pt"));
            models.addModule(classify, dropout, 160, normalMean, normalStd, "MobileNetV2");

            classify = Module.load(assetFilePath("tiny_vit_5m_224.dist_in22k_ft_in1k_full_mobile.pt"));
            dropout = Module.load(assetFilePath("tiny_vit_5m_224.dist_in22k_ft_in1k_dropout_mobile.pt"));
            models.addModule(classify, dropout, 224, imageNetMean, imageNetStd, "ViT tiny");

            classify = Module.load(assetFilePath("vit_small_patch16_224.augreg_in21k_full_mobile.pt"));
            dropout = Module.load(assetFilePath("vit_small_patch16_224.augreg_in21k_dropout_mobile.pt"));
            models.addModule(classify, dropout, 224, normalMean, normalStd, "ViT small");

            switchModel.setText(models.getCurName());

            models.loaded = true;
            runOnUiThread(() -> {
                resultText.setText("Loading done!");
            });
        });

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            openCamera();
        else
            requestAppPermissions();

        switchModel.setOnClickListener(view -> {
            if (!models.loaded)
                return;

            models.nextModule();
            switchModel.setText(models.getCurName());
        });

        loadImage.setOnClickListener(view -> {
            if (resultText.getText() == "Analyzing..." || resultText.getText() == "Loading models...")
                return;
            resultText.setText("");
            resultText2.setText("");
            oodText.setText("");
            if (!imageCaptured) {
                loadImage.setText("Take Another Image");
                new Thread(() -> {
                    runOOD(true);
                }).start();
            } else {
                loadImage.setText("Capture Image");
                readyToInfer.release();
            }
            imageCaptured = !imageCaptured;
        });

        detect.setOnClickListener(view -> {
            if (curImage != null && imageCaptured) {
                Thread t = new Thread(() -> {
                    runInference();
                });
                t.start();
            }
        });
    }

    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = {null, null, null};
    private int yRowStride = 0;
    private Runnable imageConverter;
    private Runnable postInferenceCallback;

    @Override
    public void onImageAvailable(ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            Image image = reader.acquireLatestImage();
            if (image == null)
                return;
            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            int uvRowStride = planes[1].getRowStride();
            int uvPixelStride = planes[1].getPixelStride();
            imageConverter = () -> {
                ImageUtils.INSTANCE.convertYUV420ToARGB8888(
                        yuvBytes[0],
                        yuvBytes[1],
                        yuvBytes[2],
                        previewWidth,
                        previewHeight,
                        yRowStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes
                );
            };
            postInferenceCallback = () -> {
                image.close();
                isProcessingFrame = false;
            };
            processImage();
        } catch (Exception e){
            e.printStackTrace();
            return;
        }
    }


    private void processImage() {
        imageConverter.run();
        Bitmap rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        if (imageView != null && !imageCaptured) {
            Matrix rotationMatrix = new Matrix();
            rotationMatrix.setRotate((float)sensorOrientation);
            curImage =  Bitmap.createBitmap(rgbFrameBitmap, 0, 0, rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight(), rotationMatrix, true);

            Bitmap tempBitmap = Bitmap.createBitmap(curImage.getWidth(), curImage.getHeight(), Bitmap.Config.RGB_565);
            Canvas canvas = new Canvas(tempBitmap);
            Rect dst = new Rect();
            dst.set(0, 0, curImage.getWidth(), curImage.getHeight());
            canvas.drawBitmap(curImage, null, dst, null);

//            drawRectangle(canvas, h, w);

            runOnUiThread(() -> {
                imageView.setBackground(new BitmapDrawable(getResources(), tempBitmap));
            });

            // RUN REAL TIME INFERENCE
            if (realtime.isChecked() && resultText.getText() != "Loading models...") {
                boolean acquired = readyToInfer.tryAcquire();
                if (acquired) {
                    new Thread(() -> {
                        runInference();
                        Thread t = new Thread(() -> {
                            runOOD(false);
                        });
                        t.start();
                        try {
                            t.join();
                        } catch (InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                        readyToInfer.release();
                    }).start();
                }

            }
        }
        postInferenceCallback.run();
    }

    private static void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null || yuvBytes[i].length != buffer.capacity()) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }
}