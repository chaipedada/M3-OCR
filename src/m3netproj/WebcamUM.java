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
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class WebcamUM implements KeyListener
{
    
    final String fname1 = "u.png";
    final String fname2 = "m.png";
    final int canny1 = 150;
    final int canny2 = 200;
    final int dsize = 32;
    final int iterations = 300;
        
    final int[][] locs = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {-1,1}};
    
    private boolean done = false;
    
    private int maxIterations = 40000;

    public WebcamUM()
    {
        
        System.loadLibrary("opencv_java2410");
        
        Mat t1img = Highgui.imread(fname1);
        Mat t2img = Highgui.imread(fname2);
        
        imshow(t1img, "U Training Image");
        imshow(t2img, "M Training Image");
        
        float[] td1 = generate(t1img);
        float[] td2 = generate(t2img);
        
        Hopfield hop = new Hopfield(dsize*dsize);
        hop.addTrainingData(td1);
        hop.addTrainingData(td2);
        hop.train();
        
        VideoCapture cap = new VideoCapture(0);
        if(!cap.isOpened())
            System.err.println("Couldn't open video stream");
        
        imshow(floatToMat(td1, dsize), "U Training Data");
        imshow(floatToMat(td2, dsize), "M Training Data");
        
        Mat img = new Mat(new Size(cap.get(Highgui.CV_CAP_PROP_FRAME_WIDTH), cap.get(Highgui.CV_CAP_PROP_FRAME_HEIGHT)), CvType.CV_8UC3);
        JFrame vidFrame = new JFrame("Video");
        vidFrame.setSize(img.width(), img.height());
        vidFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        vidFrame.setVisible(true);
        vidFrame.addKeyListener(this);
        try {
            done = false;
            while(!done)
            {
                cap.read(img);
                imshow(vidFrame, img);
            }
            float[] in = generate(img);
            imshow(floatToMat(in, dsize), "Input");
            
            float[] sync, async;
            
            hop.loadInputs(in);
            JFrame syncFrame = new JFrame("Sync");
            syncFrame.setSize(512, 512);
            syncFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            syncFrame.setVisible(true);
            for(int it = 0; it < maxIterations; it++)
            {
                sync = hop.recallSync(HopfieldCompleteSync.randomInts(hop.numInputs, 10));
                imshow(syncFrame, floatToMat(sync, dsize));
                if(arraysEq(sync, td1))
                {
                    JOptionPane.showMessageDialog(null, "It's a U");
                    break;
                }
                else if(arraysEq(sync, td2))
                {
                    JOptionPane.showMessageDialog(null, "It's an M");
                    break;
                }
                Thread.sleep(5);
            }
            
            hop.loadInputs(in);
            JFrame asyncFrame = new JFrame("Async");
            asyncFrame.setSize(512, 512);
            asyncFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            asyncFrame.setVisible(true);
            for(int it = 0; it < maxIterations; it++)
            {
                async = hop.recallSingle(it % hop.numInputs);
                imshow(asyncFrame, floatToMat(async, dsize));
                if(arraysEq(async, td1))
                {
                    JOptionPane.showMessageDialog(null, "It's a U");
                    break;
                }
                else if(arraysEq(async, td2))
                {
                    JOptionPane.showMessageDialog(null, "It's an M");
                    break;
                }
            }
            
        } catch(Exception e) { System.err.println(e); }
        
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
    
    public static boolean arraysEq(float[] a1, float[] a2)
    {
        if(a1.length != a2.length)
            return false;
        for(int i = 0; i < a1.length; i++)
            if(a1[i] != a2[i])
                return false;
            
        return true;
    }
    
    public static void printArray(float[] a)
    {
        for(float f : a)
            System.out.print(f + ", ");
        System.out.println();
    }
    
    public static Mat floatToMat(float[] f, int dsize)
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
        imshow(otherframe, img);
    }
    
    public static void main(String[] args)
    {
        System.loadLibrary("opencv_java2410");
        new WebcamUM();
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