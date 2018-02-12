package m3netproj;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
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
import org.opencv.imgproc.Imgproc;

public class HopfieldCompleteSync
{
    
    final String fname1 = "u.png";
    final String fname2 = "m.png";
    final int canny1 = 150;
    final int canny2 = 200;
    final int dsize = 32;
    final int maxIterations = 20000;
        
    final int[][] locs = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {-1,1}};

    public HopfieldCompleteSync()
    {
        
        System.loadLibrary("opencv_java2410");
        
        float[] td1 = generate(Highgui.imread(fname1));
        float[] td2 = generate(Highgui.imread(fname2));
        
        Hopfield hop = new Hopfield(dsize*dsize);
        hop.addTrainingData(td1);
        hop.addTrainingData(td2);
        hop.train();
        
        Mat distimg = Highgui.imread("m-dist.png");
        
        //further distort image with random pixel reassignments
        for(int i = 0; i < 1000; i++)
        {
            distimg.put((int)(Math.random() * distimg.rows()), (int)(Math.random() * distimg.cols()), (Math.random() > 0.5) ? new byte[] {0,0,0} : new byte[] {(byte)127, (byte)127, (byte)127});
        }
        imshow(distimg);
        float[] dist = generate(distimg);//generate(Highgui.imread(iCloudLatest().getAbsolutePath()));
        //for(int i = 0; i < 40; i++)
        //    dist[(int)(Math.random() * dist.length)] = (Math.random() > 0.5) ? 1f : -1f;
        
        Mat fixedmat;
        JFrame frame = new JFrame("Output");
        frame.setSize(512, 512);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        
        hop.loadInputs(dist);
        
        for(int it = 0; it < maxIterations; it++)
        {
            float[] fixed = hop.recallSync(randomInts(hop.numInputs, 30));
            fixedmat = new Mat(new Size(dsize,dsize), CvType.CV_8U);
            for(int i = 0; i < fixed.length; i++)
                fixedmat.put(i/dsize, i%dsize, new byte[] {fixed[i] == 1 ? (byte)127 : 0});
            imshow(frame, fixedmat);
            if(arraysEq(fixed, td1))
            {
                System.out.println("It's a U!!");
                break;
            }
            else if(arraysEq(fixed, td2))
            {
                System.out.println("It's an M!!");
                break;
            }
            try
            {
                Thread.sleep(5);
            } catch(Exception exc) {}
        }
        
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
                if(sum > 1000)
                {
                    data_img.put(row, col, new byte[] {(byte)127});
                    data[row*dsize + col] = 1f;
                }
                else
                    data[row*dsize + col] = -1f;
            }
        imshow(data_img);
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
    
    public static void imshow(Mat img)
    {
        JFrame otherframe = new JFrame("Original distorted image");
        otherframe.setSize((int)img.size().width, (int)img.size().height);
        otherframe.setVisible(true);
        imshow(otherframe, img);
    }
    
    public static int[] randomInts(int max, int count)
    {
        int[] ret = new int[count];
        for(int i = 0; i < count; i++)
        {
            ret[i] = (int)(Math.random() * max);
        }
        return ret;
    }
    
    public static void main(String[] args)
    {
        System.loadLibrary("opencv_java2410");
        new HopfieldCompleteSync();
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