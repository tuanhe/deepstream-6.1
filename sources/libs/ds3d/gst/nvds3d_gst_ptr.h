/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef NVDS3D_GST_NVDS3D_GST_H
#define NVDS3D_GST_NVDS3D_GST_H

#include <ds3d/common/hpp/dataloader.hpp>
#include <ds3d/common/hpp/datamap.hpp>
//#include <ds3d/common/hpp/datamap.hpp>

#include <gst/gst.h>
#include <gst/gstminiobject.h>
#include <gst/gstobject.h>

namespace ds3d { namespace gst {

struct GstObjectFunc {
    static gpointer ref(gpointer p)
    {
        DS_ASSERT(p);
        return gst_object_ref(p);
    }
    static void unref(gpointer p)
    {
        if (p) {
            return gst_object_unref(p);
        }
    }
};

template<class GstMiniObjDerived>
struct GstMiniObjectFunc {
    static GstMiniObjDerived* ref(GstMiniObjDerived* p) {
        return (GstMiniObjDerived*)gst_mini_object_ref(GST_MINI_OBJECT_CAST(p)); 
    }
    static void unref(GstMiniObjDerived* p) {
        return gst_mini_object_unref(GST_MINI_OBJECT_CAST(p));
    }
};

template <class GstObjT, class ObjFunc>
class GstPtr {
private:
    std::shared_ptr<GstObjT> _gst_obj;
    std::string _name;

public:
    GstPtr() = default;
    GstPtr(GstObjT* obj, const std::string& name = "", bool takeOwner = true)
    {
        reset(obj, takeOwner);
        setName(name);
    }
    virtual ~GstPtr() = default;

    void setName(const std::string& name) { _name = name; }

    GstPtr(GstPtr&& other)
    {
        _gst_obj = std::move(other._gst_obj);
        _name = std::move(other._name);
    }

    GstPtr(const GstPtr& other)
    {
        _gst_obj = other._gst_obj;
        _name = other._name;
    }

    GstPtr& operator=(const GstPtr& other)
    {
        _gst_obj = other._gst_obj;
        _name = other._name;
        return *this;
    }

    void reset(GstObjT* obj = nullptr, bool takeOwner = true)
    {
        GstObjT* entity = obj;
        if (!takeOwner && obj) {
            entity = (GstObjT*)ObjFunc::ref(obj);
        }
        _gst_obj.reset(entity, ObjFunc::unref);
    }

    GstObjT* copy() const
    {
        GstObjT* raw = get();
        if (raw) {
            raw = (GstObjT*)ObjFunc::ref(raw);
        }
        return raw;
    }
    const std::string& name() const { return _name; }

    operator GstObjT*() const { return get(); }

    GstObjT* get() const { return _gst_obj.get(); }
    operator bool() const { return (bool)_gst_obj; }
};

template <class GstObj>
using GstObjPtr = GstPtr<GstObj, GstObjectFunc>;
template <class GstMiniObj>
using GstMiniObjPtr = GstPtr<GstMiniObj, GstMiniObjectFunc<GstMiniObj>>;

using BusPtr = GstObjPtr<GstBus>;
using CapsPtr = GstMiniObjPtr<GstCaps>;
using BufferPtr = GstMiniObjPtr<GstBuffer>;

class PadPtr : public GstObjPtr<GstPad> {
public:
    PadPtr(GstPad* pad, bool takeOwner = true)
        : GstObjPtr<GstPad>(pad, (GST_PAD_NAME(pad) ? GST_PAD_NAME(pad) : ""), takeOwner)
    {
    }
    template <typename... Args>
    PadPtr(Args&&... args) : GstObjPtr<GstPad>(std::forward<Args>(args)...)
    {
    }
    ~PadPtr() = default;

    uint32_t addProbe(
        GstPadProbeType mask, GstPadProbeCallback callback, gpointer udata,
        GDestroyNotify destroyData)
    {
        DS_ASSERT(get());
        return gst_pad_add_probe(get(), mask, callback, udata, destroyData);
    }

    void removeProbe(uint32_t id)
    {
        DS_ASSERT(get());
        gst_pad_remove_probe(get(), id);
    }
};

class ElePtr : public GstObjPtr<GstElement> {
public:
    ElePtr(GstElement* ele, bool takeOwner = true)
        : GstObjPtr<GstElement>(
              ele, (GST_ELEMENT_NAME(ele) ? GST_ELEMENT_NAME(ele) : ""), takeOwner)
    {
    }
    template <typename... Args>
    ElePtr(Args&&... args) : GstObjPtr<GstElement>(std::forward<Args>(args)...)
    {
    }
    ~ElePtr() = default;

    PadPtr staticPad(const std::string& padName)
    {
        DS_ASSERT(get());
        PadPtr pad(gst_element_get_static_pad(get(), padName.c_str()), true);
        return pad;
    }

    ElePtr& link(ElePtr& next)
    {
        DS3D_THROW_ERROR_FMT(
            gst_element_link(get(), next.get()), ErrCode::kGst, "link element %s to %s failed",
            name().c_str(), next.name().c_str());
        return next;
    }
};

inline ElePtr
elementMake(const std::string& factoryName, const std::string& name = "")
{
    GstElement* ele = gst_element_factory_make(factoryName.c_str(), name.c_str());
    DS3D_FAILED_RETURN(
        ele, nullptr, "create element: %s, name:%s failed.", factoryName.c_str(), name.c_str());
    ElePtr ptr(ele, true);
    DS_ASSERT(ele);
    return ptr;
}

}}  // namespace ds3d::gst

#endif  // NVDS3D_GST_NVDS3D_GST_H
