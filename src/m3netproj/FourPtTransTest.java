package m3netproj;

import java.awt.Desktop;
import java.io.File;
import java.util.ArrayList;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class FourPtTransTest 
{
    
    private final int harrisThresh = 120;
    
    public FourPtTransTest()
    {
        
        System.loadLibrary("opencv_java2410");
        
        Mat img = Highgui.imread(HopfieldCompleteAsync.iCloudLatest().getAbsolutePath());
        
        Point[] pts = getCornerPts(img);
       
        //ordering
        Point[] rect = new Point[4];
        
        int[] sum = new int[4];
        for(int idx = 0; idx < 4; idx++)
            sum[idx] = (int)(pts[idx].x + pts[idx].y);
        rect[0] = pts[argmin(sum)];
        rect[2] = pts[argmax(sum)];
        
        int[] diff = new int[4];
        for(int idx = 0; idx < 4; idx++)
            sum[idx] = (int)Math.abs(pts[idx].x - pts[idx].y);
        rect[1] = pts[argmin(diff)];
        rect[3] = pts[argmax(diff)];
        
        //new image dimensions
        Point tl = rect[0];
        Point tr = rect[1];
        Point br = rect[2];
        Point bl = rect[3];

        double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
        double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));
        int maxWidth = Math.max((int)widthA, (int)widthB);
        
        double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
        double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));
        int maxHeight = Math.max((int)heightA, (int)heightB);

        Point[] dst = {new Point(0,0), new Point(maxWidth - 1, 0),
            new Point(maxWidth-1, maxHeight-1), new Point(0, maxHeight-1)};
        
        Mat srcmat = new Mat(new Size(4,1), CvType.CV_32FC2);
        Mat dstmat = new Mat(new Size(4,1), CvType.CV_32FC2);
        for(int i = 0; i < 4; i++)
        {
            srcmat.put(i, 0, new float[] {(float)rect[i].x, (float)rect[i].y});
            dstmat.put(i, 0, new float[] {(float)dst[i].x, (float)dst[i].y});
        }
     
        Mat m = Imgproc.getPerspectiveTransform(srcmat, dstmat);
        Mat warped = new Mat(img.size(), img.type());
        Imgproc.warpPerspective(img, warped, m, new Size(maxWidth, maxHeight));
        
        //saveAndOpen(warped, "warped.png");
        
    }
    
    private Point[] getCornerPts(Mat img)
    {
        //filtering
        Mat filt = new Mat(img.size(), img.type());
        Core.inRange(img, new Scalar(0,0,0), new Scalar(100,100,255), filt); //filter bg
        Imgproc.erode(filt, filt, Mat.ones(new Size(5,5), CvType.CV_8U)); //open
        Imgproc.dilate(filt, filt, Mat.ones(new Size(6,6), CvType.CV_8U));
        
        //detect corners
        Mat corners = new Mat(filt.size(), filt.type());
        ArrayList<Point> clist = new ArrayList<>();
        Imgproc.cornerHarris(filt, corners, 2, 3, 0.04);
        Core.normalize(corners, corners, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        for(int j = 0; j < corners.rows(); j++)
            for(int i = 0; i < corners.cols(); i++)
               if(corners.get(j, i)[0] > harrisThresh)
                    clist.add(new Point(i,j));
        //find outside 4
        Point[] rect = {new Point(10000, 10000), new Point(-10000, 10000), new Point(-10000, -10000), new Point(10000, -10000)};
        for(Point c : clist)
        {
            if(c.x < rect[0].x && c.y < rect[0].y)
                rect[0] = c;
            if(c.x > rect[1].x && c.y < rect[1].y)
                rect[1] = c;
            if(c.x > rect[2].x && c.y > rect[2].y)
                rect[2] = c;
            if(c.x < rect[3].x && c.y > rect[3].y)
                rect[3] = c;
        }
        
        for(Point p : rect)
        {
            Core.circle(filt, p, 10, new Scalar(180), 3);
        }
        
        saveAndOpen(filt, "filt.png");
        
        return rect;
        
    }
    
    private int argmin(int[] array)
    {
        int smallest = Integer.MAX_VALUE, idx = 0;
        for(int i = 0; i < array.length; i++)
            if(array[i] < smallest)
            {
                smallest = array[i];
                idx = i;
            }
        return idx;
    }
    
    private int argmax(int[] array)
    {
        int largest = Integer.MIN_VALUE, idx = 0;
        for(int i = 0; i < array.length; i++)
            if(array[i] > largest)
            {
                largest = array[i];
                idx = i;
            }
        return idx;
    }
    
    private void saveAndOpen(Mat m, String name)
    {
        Highgui.imwrite(name, m);
        try
        {
            Desktop.getDesktop().open(new File(name));
        } catch(Exception e) {}
    }
    
    public static void main(String[] args)
    {
        new FourPtTransTest();
    }
    
}
