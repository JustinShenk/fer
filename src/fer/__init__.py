#!/usr/bin/python3

# MIT License
#
# Copyright (c) 2018 Justin Shenk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging

from .classes import Video
from .fer import FER

log = logging.getLogger("fer")
log.setLevel(logging.INFO)

__version__ = "25.10.2"

__title__ = "fer"
__description__ = "Facial expression recognition from images"
__url__ = "https://github.com/justinshenk/fer"
__uri__ = __url__
__doc__ = __description__ + " <" + __url__ + ">"

__author__ = "Justin Shenk"
__email__ = "shenkjustin@gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2019 " + __author__
