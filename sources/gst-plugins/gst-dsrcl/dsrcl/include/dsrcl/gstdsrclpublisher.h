/* GStreamer
 * Copyright (C) 2022 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_DSRCLPUBLISHER_H_
#define _GST_DSRCLPUBLISHER_H_

//#include "ros2_deepstream_msgs/msg/nv_ds_meta_data.hpp"

#include <memory>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "rclcpp/rclcpp.hpp"
#include "dsmsg/msg/num.hpp"  


/* Package and library details required for plugin_init */
#define PACKAGE "ds_rcl_publisher"
#define VERSION "1.0"
#define LICENSE "LGPL"
#define DESCRIPTION "Publish NVDS meta data to a ROS2 topic"
#define BINARY_PACKAGE "Publish NVDS meta data to a ROS2 topic"
#define URL "https://github.com/tuanhe"

G_BEGIN_DECLS

typedef struct _GstDsRclPublisher GstDsRclPublisher;
typedef struct _GstDsRclPublisherClass GstDsRclPublisherClass;

/* Standard boilerplate stuff */
#define GST_TYPE_DSRCLPUBLISHER   (gst_dsrclpublisher_get_type())
#define GST_DSRCLPUBLISHER(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSRCLPUBLISHER,GstDsRclPublisher))
#define GST_DSRCLPUBLISHER_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSRCLPUBLISHER,GstDsRclPublisherClass))
#define GST_DSRCLPUBLISHER_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj),GST_TYPE_DSRCLPUBLISHER,GstDsRclPublisherClass))
#define GST_IS_DSRCLPUBLISHER(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSRCLPUBLISHER))
#define GST_IS_DSRCLPUBLISHER_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSRCLPUBLISHER))
#define GST_DSRCLPUBLISHER_CAST(obj)  ((GstDsRclPublisher *)(obj))

class GstDsRclPublisherNode : public rclcpp::Node {
public:
  GstDsRclPublisherNode(const std::string& name, const std::string& topic_name);
  ~GstDsRclPublisherNode();

  static void Process(GstDsRclPublisherNode* node, const NvDsFrameMeta* data);

private:
  //void Publish(ros2_deepstream_msgs::msg::NvDsMetaData::UniquePtr msg);
  //std::shared_ptr<rclcpp::Publisher<NvDsMetaData>> string_publisher;
  std::shared_ptr<rclcpp::Publisher<dsmsg::msg::Num>> string_publisher;
};

struct _GstDsRclPublisher
{
  GstBaseTransform base_trans;

  // Context of the custom algorithm library
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

struct _GstDsRclPublisherClass
{
  GstBaseTransformClass base_dsrclpublisher_class;
};

GType gst_dsrclpublisher_get_type (void);

G_END_DECLS

#endif