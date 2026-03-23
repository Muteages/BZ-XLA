# Build TensorFlow Serving

## Conda

``` bash
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
```

## Python

``` bash
conda create -n serving python=3.9
conda activate serving
```

## Bazel

Since we’re working with TF 2.20, please use __bazel 7.4.1__

``` bash
# Option 1:
conda install bazel=7.4.1

# Option 2:
wget "https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-linux-arm64" --no-check-certificate
```

## JDK & C/C++ compiler

They have already been pre-installed on the server, so we can use it directly.

## Automake

The build might fail because automake is missing, so it may need to be installed manually.

``` bash
conda install -c conda-forge -y automake autoconf libtool pkg-config make cmake
```

## tf-serving

``` bash
git clone https://github.com/tensorflow/serving.git
git checkout 1c887668
```

## tensorflow

``` bash
git clone git@github.com:your_account_name/tensorflow.git
```

``` bash
 --override_repository="org_tensorflow=your_tensorflow_path"
```


## boost

``` bash
cd serving
mkdir -p dist && cd dist
git clone https://github.com/boostorg/boost.git
git checkout b7b1371294b4bdfc8d85e49236ebced114bc1d8f
git submodule update --init --recursive
```

Apply this patch to serving

```
diff --git a/tensorflow_serving/workspace.bzl b/tensorflow_serving/workspace.bzl
index de1203a7..7af4fdd7 100644
--- a/tensorflow_serving/workspace.bzl
+++ b/tensorflow_serving/workspace.bzl
@@ -123,11 +123,8 @@ def tf_serving_workspace():

     # The Boost repo is organized into git sub-modules (see the list at
     # https://github.com/boostorg/boost/tree/master/libs), which requires "new_git_repository".
-    new_git_repository(
-        name = "org_boost",
-        commit = "b7b1371294b4bdfc8d85e49236ebced114bc1d8f",  # boost-1.75.0
-        build_file = "//third_party/boost:BUILD",
-        init_submodules = True,
-        recursive_init_submodules = True,
-        remote = "https://github.com/boostorg/boost",
+    native.new_local_repository(
+       name="org_boost",
+       build_file="//third_party/boost:BUILD",
+       path="./dist/org_boost",
     )
```

## Build Cmd

``` 
bazel --output_user_root=./output \
 build -c opt \
 --distdir=$HOME/serving/dist tensorflow_serving/model_servers:tensorflow_model_server \
 --override_repository="org_tensorflow=your_tensorflow_path"
```

