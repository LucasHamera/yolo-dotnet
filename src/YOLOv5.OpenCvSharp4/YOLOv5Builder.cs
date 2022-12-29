using OpenCvSharp.Dnn;

namespace YOLOv5.OpenCvSharp4
{
    public class YOLOv5Builder
    {
        public IYOLOv5 Build(string weightPath, Target target = default, Backend backend = default)
        {
            var net = Net.ReadNet(weightPath);
            net.SetPreferableTarget(Target.CUDA);
            net.SetPreferableBackend(Backend.CUDA);
            return new YOLOv5(net);
        }
    }
}
