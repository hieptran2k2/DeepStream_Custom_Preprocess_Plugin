/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file
 * <b>NVIDIA GStreamer DeepStream: Custom Events</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer custom
 * event functions, used to map events to individual sources which
 * are batched together by Gst-nvstreammux.
 *
 */

/**
 * @defgroup  gstreamer_nvevent  Events: Custom Events API
 *
 * Specifies GStreamer custom event functions, used to map events
 * to individual sources which are batched together by Gst-nvstreammux.
 *
 * @ingroup gst_mess_evnt_qry
 * @{
 */

#ifndef __GST_NVDSCUSTOMEVENT_H__
#define __GST_NVDSCUSTOMEVENT_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif


/** Defines Roi structure for GST_NVCUSTOMEVENT_ROI_UPDATE custom event */
  typedef struct RoiDimension
  {
    gchar roi_id[128];
    guint left;
    guint top;
    guint width;
    guint height;
  } RoiDimension;

#define FLAG(name) GST_EVENT_TYPE_##name

/** Defines supported types of custom events. */
  typedef enum
  {
  /** Specifies a custom event to indicate ROI update for preprocess
   of a particular stream in a batch. */
    GST_NVEVENT_ROI_UPDATE
        = GST_EVENT_MAKE_TYPE (406, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event to indicate infer interval update
   of a particular stream in a batch. */
    GST_NVEVENT_INFER_INTERVAL_UPDATE
        = GST_EVENT_MAKE_TYPE (407, FLAG (DOWNSTREAM) | FLAG (SERIALIZED)),
  /** Specifies a custom event to indicate osd process mode update
   of a particular stream in a batch. */
    GST_NVEVENT_OSD_PROCESS_MODE_UPDATE
        = GST_EVENT_MAKE_TYPE (408, FLAG (DOWNSTREAM) | FLAG (SERIALIZED))
  } GstNvDsCustomEventType;
#undef FLAG

/**
 * Creates a new "roi-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which nv-roi-update is to be sent
 * @param[out] roi_count    The roi_count obtained corresponding to stream ID for the event.
 * @param[out] roi_dim      The RoiDimension structure of size roi_count.
 */
  GstEvent *gst_nvevent_new_roi_update (gchar * stream_id, guint roi_count,
      RoiDimension * roi_dim);

/**
 * Parses a "roi-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a nv-roi-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] roi_count    A pointer to the parsed number of roi(s)
 *                          corresponding to stream ID for the event.
 * @param[out] roi_dim      A double pointer to the parsed RoiDimension structure of size roi_count.
 *                          User MUST free roi_dim memory using g_free post usage.
 */
  void gst_nvevent_parse_roi_update (GstEvent * event, gchar ** stream_id,
      guint * roi_count, RoiDimension ** roi_dim);

/**
 * Creates a new "nv-infer-interval-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which infer-interval-update is to be sent
 * @param[out] interval     The infer interval obtained corresponding to stream ID for the event.
 */
  GstEvent *gst_nvevent_infer_interval_update (gchar * stream_id,
      guint interval);

/**
 * Parses a "nv-infer-interval-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a infer-interval-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] interval     A pointer to the parsed interval
 *                          corresponding to stream ID for the event.
 */
  void gst_nvevent_parse_infer_interval_update (GstEvent * event,
      gchar ** stream_id, guint * interval);


/**
 * Creates a new "nv-osd-process-mode-update" event.
 *
 * @param[out] stream_id    Stream ID of the stream for which osd-process-mode-update is to be sent
 * @param[out] process_mode The infer interval obtained corresponding to stream ID for the event.
 */
  GstEvent *gst_nvevent_osd_process_mode_update (gchar * stream_id,
      guint process_mode);

/**
 * Parses a "nv-osd-process-mode-update" event received on the sinkpad.
 *
 * @param[in] event         The event received on the sinkpad
 *                          when the stream ID sends a osd-process-mode-update event.
 * @param[out] stream_id    A pointer to the parsed stream ID for which
 *                          the event is sent.
 * @param[out] process_mode A pointer to the parsed interval
 *                          corresponding to stream ID for the event.
 */
  void gst_nvevent_parse_osd_process_mode_update (GstEvent * event,
      gchar ** stream_id, guint * process_mode);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
