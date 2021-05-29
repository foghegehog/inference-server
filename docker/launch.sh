arg_tag=tensorrt-ubuntu-1804
arg_gpus=all
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --tag) arg_tag="$2"; shift;;
  --gpus) arg_gpus="$2"; shift;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help or -h         : Print this help menu."
    echo " --tag     <imagetag> : Image name for generated container."
    echo " --gpus    <number>   : Number of GPUs visible in container. Set 'none' to disable, and 'all' to make all visible."
    exit;
fi

extra_args=""
if [ "$arg_gpus" != "none" ]; then
    extra_args="$extra_args --gpus $arg_gpus"
fi

docker_args="$extra_args -v ${PWD}:/workspace/TensorRT -v /usr/src/tensorrt/data:/workspace/data -p 0.0.0.0:8080:8080 --rm -it $arg_tag:latest"

echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
