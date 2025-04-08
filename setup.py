import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def run(self):
        tf_cflags = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))',
                ]
            )
            .decode()
            .strip()
            .split()
        )

        tf_lflags = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))',
                ]
            )
            .decode()
            .strip()
            .split()
        )

        cmd = (
            [
                "g++",
                "-std=c++14",
                "-shared",
                "src/deform_conv2d.cpp",
                "src/deform_conv2d_kernel.cpp",
                "-o",
                "src/deform_conv2d.so",
                "-fPIC",
                "-O2",
                "-Wno-deprecated-declarations",
            ]
            + tf_cflags
            + tf_lflags
        )

        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]

        subprocess.check_call(cmd)
        super().run()


setup(
    name="deform_conv2d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["tensorflow"],
    cmdclass={"build_ext": CustomBuildExt},
    package_data={"": ["deform_conv2d.so"]},
    include_package_data=True,
    zip_safe=False,
)
