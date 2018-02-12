package m3netproj;

import java.io.File;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class DistortedNumberDetect
{
    
    final int canny1 = 150;
    final int canny2 = 200;
    final int dsize = 32;
    final int iterations = 1000;
        
    final int[][] locs = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {-1,1}};

    public DistortedNumberDetect()
    {
        
        System.loadLibrary("opencv_java2410");
        
        float[][] td = {generate(Highgui.imread("num_training\\2.png")), generate(Highgui.imread("num_training\\3.jpg"))};

        Hopfield hop = new Hopfield(dsize*dsize);
        for(float[] t : td)
        {
            hop.addTrainingData(t);
            WebcamUM.imshow(WebcamUM.floatToMat(t, dsize), "Training Data");
        }
        hop.train();
        
        Mat distimg = Highgui.imread("num_training\\3.jpg");
        for(int i = 0; i < 1000; i++)
        {
            distimg.put((int)(Math.random()*distimg.rows()), (int)(Math.random()*distimg.cols()), new byte[] {127,127,127});
        }
        WebcamUM.imshow(distimg, "Distorted Input");
        float[] dist = generate(distimg);//generate(Highgui.imread(iCloudLatest().getAbsolutePath()));
        float[] fixed = hop.recall(dist, iterations);

        printArray(fixed);
        WebcamUM.imshow(WebcamUM.floatToMat(fixed, dsize), "Result");

        int res = -1;
        for(int i = 0; i < td.length; i++)
            if(arraysEq(td[i], fixed))
                res = i;
        if(res == -1)
            JOptionPane.showMessageDialog(null, "No idea what that is");
        else
            JOptionPane.showMessageDialog(null, "It's a " + (res+2));
        
    }
    
    private float[] generate(Mat src)
    {        
        Mat pre = preprocess(src);
        Mat data_img = Mat.zeros(new Size(dsize, dsize), CvType.CV_8U);
        float[] data = new float[dsize*dsize];
        
        //create dsize*dsize version
        int w = pre.cols() / data_img.cols();
        int h = pre.rows() / data_img.rows();
        for(int row = 0; row < dsize; row++)
            for(int col = 0; col < dsize; col++)
            {
                double sum = Core.sumElems(pre.submat(new Rect(col*w, row*h, w, h))).val[0];
                if(sum > 1500)
                {
                    data_img.put(row, col, new byte[] {(byte)127});
                    data[row*dsize + col] = 1f;
                }
                else
                    data[row*dsize + col] = -1f;
            }
        return data;
    }
    
    /** Apply canny */
    private Mat preprocess(Mat img)
    {
        Mat img_gray = new Mat(img.size(), img.type());
        Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_BGR2GRAY);
        
        Mat canny = new Mat(img.size(), img.type());
        Imgproc.Canny(img_gray, canny, canny1, canny2);        

        return canny;
    }
    
    private boolean arraysEq(float[] a1, float[] a2)
    {
        if(a1.length != a2.length)
            return false;
        for(int i = 0; i < a1.length; i++)
            if(a1[i] != a2[i])
                return false;
            
        return true;
    }
    
    private void printArray(float[] a)
    {
        for(float f : a)
            System.out.print(f + ", ");
        System.out.println();
    }
    
    public static File iCloudLatest()
    {
        File[] files = new File("C:\\Users\\Benjamin\\Pictures\\iCloud Photos\\My Photo Stream").listFiles();
        long mod = 0L;
        File latest = null;
        for (File file : files)
            if (file.lastModified() > mod) 
            {
                latest = file;
                mod = file.lastModified();
            }
        return latest;
    }
    
    public static void main(String[] args)
    {
        System.loadLibrary("opencv_java2410");
        new DistortedNumberDetect();
    }
    
    /*
    OLD CRAP
    
    for(int pos = 0; pos < 8; pos++) //relative method
    try
    {
        if(harris.get(j, i)[0] 
             - harris.get(j + locs[pos][0], i + locs[pos][1])[0] > harrisThresh)
             Core.circle(ret, new Point(i,j), 0, new Scalar(255));
    } catch(Exception e) {}
    
    
    
    Imgproc.cornerHarris(img_gray, harris, harrisBlockSize, harrisAperatureSize, harrisK);
    Core.normalize(harris, harris, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
    for(int j = 0; j < harris.rows(); j++)
            for(int i = 0; i < harris.cols(); i++)
                if(harris.get(i, j)[0] > harrisThresh)
                {
                    Core.circle(ret, new Point(j,i), 0, new Scalar(255));
                    corners.add(new Point(j,i));
                }
        
    
    */
    
}