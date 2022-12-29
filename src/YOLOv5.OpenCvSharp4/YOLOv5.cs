using YOLO.Models;
using OpenCvSharp;
using System.Drawing;
using OpenCvSharp.Dnn;

namespace YOLOv5.OpenCvSharp4
{
    internal class YOLOv5 : IYOLOv5
    {
        private const int NetWidth = 640;
        private const int NetHeight = 640;
        private const double ScaleFactor = 1.0 / 255;
        private const int ClassesOffset = 5;
        private const string NetOutputLayerName = "output";

        private readonly Net _net;

        public YOLOv5(Net net)
        {
            _net = net;
        }

        public IEnumerable<DetectedObject> Forward(Mat image, float confidenceThreshold, float nmsThreshold)
        {
            using var resizedImage = ResizeImage(image);
            using var blob = CvDnn.BlobFromImage(
                resizedImage,
                ScaleFactor,
                new OpenCvSharp.Size(NetWidth, NetHeight),
                Scalar.All(0),
                true,
                false
            );

            _net.SetInput(blob);
            using var output = _net.Forward(NetOutputLayerName);

            return ConvertOutput(resizedImage, output, confidenceThreshold, nmsThreshold);
        }

        private Mat ResizeImage(Mat image)
        {
            var col = image.Cols;
            var row = image.Rows;
            var _max = Math.Max(col, row);
            Mat resized = Mat.Zeros(_max, _max, MatType.CV_8UC3);
            image.CopyTo(resized[new Rect(0, 0, col, row)]);
            return resized;
        }

        private IEnumerable<DetectedObject> ConvertOutput(Mat inputImage, Mat output, float confidenceThreshold, float nmsThreshold)
        {
            var dims = output.Dims;
            if (dims < 3)
                return Enumerable.Empty<DetectedObject>();

            var sizes = Enumerable
                .Range(0, dims)
                .Select(output.Size)
                .ToArray();

            var x_factor = 1.0f * inputImage.Cols / NetWidth;
            var y_factor = 1.0f * inputImage.Rows / NetHeight;

            var classes = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect>();

            unsafe
            {
                var dataPtr = (float*)output.DataPointer;
                var rows = sizes[1];
                for (var i = 0; i < rows; i++, dataPtr += sizes[2])
                {
                    var confidence = dataPtr[4];
                    if (confidence < confidenceThreshold)
                        continue;

                    var scorePtr = dataPtr + ClassesOffset;
                    var classCount = sizes[2] - ClassesOffset;
                    using var scores = new Mat(1, classCount, MatType.CV_32FC1, new IntPtr(scorePtr));
                    Cv2.MinMaxLoc(scores, out _, out OpenCvSharp.Point max);
                    var @class = max.X;
                    var probability = scores.At<float>(0, @class);

                    if (probability < confidenceThreshold)
                        continue;

                    var cx = dataPtr[0];
                    var cy = dataPtr[1];
                    var w = dataPtr[2];
                    var h = dataPtr[3];

                    var left = (int)((cx - 0.5 * w) * x_factor);
                    var top = (int)((cy - 0.5 * h) * y_factor);
                    var width = (int)(w * x_factor);
                    var height = (int)(h * y_factor);

                    classes.Add(@class);
                    confidences.Add(confidence);
                    probabilities.Add(probability);
                    boxes.Add(new Rect(left, top, width, height));
                }
            }

            CvDnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, out int[] indices);

            return indices
                .Select(i => new DetectedObject(classes[i], confidences[i], probabilities[i], ToRectangle(boxes[i])))
                .ToArray();
        }

        private Rectangle ToRectangle(Rect rect)
            => new Rectangle(rect.X, rect.Y, rect.Width, rect.Height);

        public void Dispose()
        {
            _net?.Dispose();
        }
    }
}