/**
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __GST_dsrclpublisher_H__
#define __GST_dsrclpublisher_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

/* Open CV headers */
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#ifdef WITH_OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#endif
#pragma GCC diagnostic pop

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "rclcpp/rclcpp.hpp"
#include "dsmsg/msg/num.hpp"  
//#include "dsexample_lib/dsexample_lib.h"

/* Package and library details required for plugin_init */
#define PACKAGE "hubin dsrclpublisher"
#define VERSION "0.9"
#define LICENSE "LGPL"
#define DESCRIPTION "hubin NVIDIA example plugin for integration with DeepStream on DGPU"
#define BINARY_PACKAGE "hubin NVIDIA DeepStream 3rdparty IP integration example plugin"
#define URL "https://github.com/tuanhe"


G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstDsRclPublisher GstDsRclPublisher;
typedef struct _GstDsRclPublisherClass GstDsRclPublisherClass;

/* Standard boilerplate stuff */
#define GST_TYPE_DSRCLPUBLISHER (gst_dsrclpublisher_get_type())
#define GST_DSRCLPUBLISHER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSRCLPUBLISHER,GstDsRclPublisher))
#define GST_DSRCLPUBLISHER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSRCLPUBLISHER,GstDsRclPublisherClass))
#define GST_DSRCLPUBLISHER_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSRCLPUBLISHER, GstDsRclPublisherClass))
#define GST_IS_dsrclpublisher(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSRCLPUBLISHER))
#define GST_IS_dsrclpublisher_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSRCLPUBLISHER))
#define GST_DSRCLPUBLISHER_CAST(obj)  ((GstDsRclPublisher *)(obj))

class GstDsRclPublisherNode : public rclcpp::Node {
public:
  GstDsRclPublisherNode(const std::string& name, const std::string& topic_name);
  ~GstDsRclPublisherNode();

  static void Process(GstDsRclPublisherNode* node, const NvDsFrameMeta* data);

private:
  //void Publish(ros2_deepstream_msgs::msg::NvDsMetaData::UniquePtr msg);
  //std::shared_ptr<rclcpp::Publisher<NvDsMetaData>> string_publisher;
public:
  std::shared_ptr<rclcpp::Publisher<dsmsg::msg::Num>> publisher;
};

struct _GstDsRclPublisher
{
  GstBaseTransform base_trans;

  // Context of the custom algorithm library
  //DsExampleCtx *dsrclpublisherlib_ctx;
  std::unique_ptr<GstDsRclPublisherNode> node;

  // Unique ID of the element. The labels generated by the element will be
  // updated at index `unique_id` of attr_info array in NvDsObjectParams.
  guint unique_id;

  // Frame number of the current input buffer
  guint64 frame_num;

  // CUDA Stream used for allocating the CUDA task
  cudaStream_t cuda_stream;

  // Host buffer to store RGB data for use by algorithm
  void *host_rgb_buf;

  // the intermediate scratch buffer for conversions RGBA
  NvBufSurface *inter_buf;

  // Input video info (resolution, color format, framerate, etc)
  GstVideoInfo video_info;

  // Resolution at which frames/objects should be processed
  gint processing_width;
  gint processing_height;

  // Flag which defince igpu/dgpu
  guint is_integrated;

  // Amount of objects processed in single call to algorithm
  guint batch_size;

  // GPU ID on which we expect to execute the task
  guint gpu_id;

  // Boolean indicating if entire frame or cropped objects should be processed
  gboolean process_full_frame;

  // Boolean indicating if to blur the detected objects
  gboolean blur_objects;
};

// Boiler plate stuff
struct _GstDsRclPublisherClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsrclpublisher_get_type (void);

G_END_DECLS
#endif /* __GST_DSRCLPUBLISHER_H__ */
