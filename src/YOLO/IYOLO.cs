using YOLO.Models;

namespace YOLO
{
    public interface IYOLO<in TInputImage> where TInputImage : class
    {
        IEnumerable<DetectedObject> Forward(TInputImage image, double confidenceThreshold, double nmsThreshold);
    }
}