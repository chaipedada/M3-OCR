package m3netproj;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class NumberRec implements KeyListener
{
    
    final int canny1 = 150;
    final int canny2 = 200;
    final int dsize = 24;
    final int iterations = 300;
        
    final int[][] locs = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {-1,1}};
    
    private boolean done = false;
    
    private int maxIterations = 10000;
    
    private int[] exts = {1, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    private String[] extsStr = {".jpg", ".png"};

    public NumberRec()
    {
        
        System.loadLibrary("opencv_java2410");

        Hopfield hop = new Hopfield(dsize*dsize);
        for(int i = 0; i < 4; i++) //change to choose which nums to upload
            hop.addTrainingData(generate(Highgui.imread("num_training\\" + i + extsStr[exts[i]])));
        hop.train();
        
        float[] one = generate(Highgui.imread("num_training\\1.jpg")); //image to try to recall
        WebcamUM.imshow(WebcamUM.floatToMat(one, dsize), "Original");
        float[] ret = hop.recall(one, iterations);
        WebcamUM.imshow(WebcamUM.floatToMat(ret, dsize), "Recalled");
        
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
                double[] sumparts = Core.sumElems(pre.submat(new Rect(col*w, row*h, w, h))).val;
                double sum = sumparts[0] + sumparts[1] + sumparts[2];
                System.out.println(sum);
                if(sum > 0)
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
    
    public Mat floatToMat(float[] f)
    {
        Mat mat = new Mat(new Size(dsize,dsize), CvType.CV_8U);
        for(int i = 0; i < f.length; i++)
            mat.put(i/dsize, i%dsize, new byte[] {f[i] == 1 ? (byte)127 : 0});
        return mat;
    }
    
    public static void imshow(JFrame frame, Mat img)
    {
        Mat resized = new Mat(img.size(), img.type());
        Imgproc.resize(img, resized, new Size(768,768), 0, 0, Imgproc.INTER_NEAREST);
        
        MatOfByte mob = new MatOfByte();
        Highgui.imencode(".jpg", resized, mob);
        byte[] bytearray = mob.toArray();
        
        BufferedImage bimg;
        try
        {
            InputStream in = new ByteArrayInputStream(bytearray);
            bimg = ImageIO.read(in);
            try
            {
                frame.getContentPane().remove(0);
            } catch(ArrayIndexOutOfBoundsException e) {}
            frame.getContentPane().add(new JLabel(new ImageIcon(bimg)));
            frame.pack();
        }
        catch(Exception e) { System.err.println(e); }
    }
    
    public static void imshow(Mat img, String title)
    {
        JFrame otherframe = new JFrame(title);
        otherframe.setSize((int)img.size().width, (int)img.size().height);
        otherframe.setVisible(true);
        otherframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        imshow(otherframe, img);
    }
    
    public static void main(String[] args)
    {
        System.loadLibrary("opencv_java2410");
        new NumberRec();
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

    @Override
    public void keyTyped(KeyEvent e) 
    {
        done = true;
    }

    @Override
    public void keyPressed(KeyEvent e) {}

    @Override
    public void keyReleased(KeyEvent e) {}
    
}