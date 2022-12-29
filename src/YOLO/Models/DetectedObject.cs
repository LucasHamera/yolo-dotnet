using System.Drawing;

namespace YOLO.Models
{
    public record struct DetectedObject(int Class, double Confidence, double Probability, Rectangle Rectangle);
}
