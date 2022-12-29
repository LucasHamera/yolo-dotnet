using OpenCvSharp;
using YOLO.Models;
using System.Drawing;

namespace YOLOv5.OpenCvSharp4.Tests
{
    public class YOLOv5lTests: IDisposable
    {
        [Fact]
        public void GivingBusImageShouldDetect4PeopleAndBus()
        {
            var imageName = "bus.jpg";
            using var image = ReadImage(imageName);

            var detectedObjects = _yolo.Forward(image, ConfidenceThreshold, NMSThreshold);

            detectedObjects
                .Should()
                .BeEquivalentTo(DetectedObjectForBusImage);
        }

        [Fact]
        public void GivingZidaneImageShouldDetect2PeopleAnd2Ties()
        {
            var imageName = "zidane.jpg";
            using var image = ReadImage(imageName);

            var detectedObjects = _yolo.Forward(image, ConfidenceThreshold, NMSThreshold);

            detectedObjects
                .Should()
                .BeEquivalentTo(DetectedObjectForZidaneImage);
        }

        private float ConfidenceThreshold => 0.25f;
        private float NMSThreshold => 0.45f;

        private IYOLOv5 BuildYOLO()
        {
            var builder = new YOLOv5Builder();
            return builder.Build("yolov5l.onnx");
        }

        private Mat ReadImage(string path)
            => Cv2.ImRead(path);
        
        private IEnumerable<DetectedObject> DetectedObjectForBusImage =>
            new[]
            {
                new DetectedObject(5, 0.9342213f, 0.9854489f, new Rectangle(21, 231, 780, 499)),
                new DetectedObject(0, 0.91989124f, 0.99633247f, new Rectangle(50, 397, 196, 510)),
                new DetectedObject(0, 0.918375f, 0.99791056f, new Rectangle(668, 395, 141, 487)),
                new DetectedObject(0, 0.90346867f, 0.96766436f, new Rectangle(224, 406, 119, 454)),
                new DetectedObject(0, 0.7906605f, 0.98915464f, new Rectangle(0, 550, 78, 323))
            };

        private IEnumerable<DetectedObject> DetectedObjectForZidaneImage =>
            new[]
            {
                new DetectedObject(0, 0.9446051f, 0.99506336f, new Rectangle(748, 42, 388, 668)),
                new DetectedObject(0, 0.8733639f, 0.99803096f, new Rectangle(132, 200, 981, 511)),
                new DetectedObject(27, 0.79220307f, 0.98892564f, new Rectangle(435, 436, 87, 280)),
                new DetectedObject(27, 0.3549678f, 0.9823775f, new Rectangle(962, 297, 60, 121))
                };

        public void Dispose()
        {
            _yolo.Dispose();
        }

        private readonly IYOLOv5 _yolo;
        public YOLOv5lTests()
        {
            _yolo = BuildYOLO();
        }
    }
}