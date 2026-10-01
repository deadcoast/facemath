from dataclasses import replace
from pathlib import Path
import copy
import json

import cv2
import numpy as np
import pytest

from facemath.observation.acquisition import (
    AcquisitionPolicy, AcquisitionRule, ApertureSource, MeasuredGroup, Predicate,
    FEATURE_NAMES, AGGREGATES, UNMATCHED, AMBIGUOUS,
    measure_source, write_group, load_group, group_from_spec, policy_from_document,
)
from facemath.observation.acquisition_lab import demonstration_policy, APERTURE, measured_record
from facemath.observation.contracts import ObservationError, array_digest, object_digest
from facemath.observation.study import read_json, file_inventory, save_rgb


def policy(*predicates, maximum=2):
    return AcquisitionPolicy('prespecified numerical test',(
        AcquisitionRule('inside',tuple(predicates) or (Predicate('min.native_short_edge_px','>=',4),)),),maximum_group_members=maximum)


def source(value=128,shape=(40,60,3),lineage='source'):
    return ApertureSource(np.full(shape,value,np.uint8),(lineage,))


def rehash(root):
    m=read_json(root/'manifest.json');m.pop('content_sha256')
    m['files']=file_inventory(root,exclude=('manifest.json',));m['content_sha256']=object_digest(m)
    (root/'manifest.json').write_text(json.dumps(m))


def test_constant_image_has_zero_gradient_range_and_correct_code():
    f,g=measure_source(source(64),policy())
    assert f[0]==40 and f[1]==2400
    np.testing.assert_allclose(f[2:5],64/255,atol=1e-15)
    assert f[5]==0 and f[6]==0 and f[7]==0 and f[8]==0
    assert g.shape==(40,60)


@pytest.mark.parametrize('value,low,high',[(0,1,0),(1,1,0),(2,0,0),(253,0,0),(254,0,1),(255,0,1)])
def test_native_code_tail_thresholds_are_inclusive(value,low,high):
    f,_=measure_source(source(value),policy())
    assert f[6]==low and f[7]==high


def test_gray_coefficients_are_explicit_rgb_not_bgr():
    im=np.zeros((16,16,3),np.uint8);im[:,:,0]=255
    f,g=measure_source(ApertureSource(im,('red',)),policy())
    np.testing.assert_allclose(g,.299)
    assert f[2]==pytest.approx(.299)


def test_sobel_energy_matches_independent_spatial_kernel_calculation():
    im=np.random.default_rng(10).integers(1,250,(11,13,3),dtype=np.uint8)
    p=policy();f,g=measure_source(ApertureSource(im,('kernel',)),p)
    kx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8;ky=kx.T
    patches=np.lib.stride_tricks.sliding_window_view(g,(3,3))
    dx=(patches*kx).sum((-1,-2));dy=(patches*ky).sum((-1,-2))
    expected=np.mean(dx*dx+dy*dy)/(np.var(g[1:-1,1:-1])+p.variance_floor)
    assert f[-1]==pytest.approx(expected,rel=1e-12,abs=1e-14)


def test_area_reduction_matches_its_frozen_grid_and_does_not_upsample():
    im=np.random.default_rng(8).integers(0,256,(256,128,3),dtype=np.uint8)
    s=ApertureSource(im,('resize',));f,g=measure_source(s,policy())
    expected=cv2.resize((im.astype(float)@np.array([299.,587.,114.]))/255000.,(64,128),interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(g,expected)
    assert f[0]==128 and f[1]==32768
    small=ApertureSource(im,('small',),(0,0,8,8))
    assert measure_source(small,policy())[1].shape==(8,8)


def test_extreme_aperture_does_not_silently_change_aspect_ratio():
    with pytest.raises(ObservationError,match='aspect ratio'):
        measure_source(ApertureSource(np.zeros((8,4000,3),np.uint8),('extreme',)),policy())


@pytest.mark.parametrize('aperture',[(-1,0,8,8),(0,0,61,8),(0,0,3,8),(0,0,8,3),(5,0,5,10),
    (0.,0,8,8),(False,0,8,8),(0,0,8),(0,0,8,8,8),(0,0,-1,4)])
def test_invalid_apertures_are_rejected_not_clipped(aperture):
    with pytest.raises(ObservationError):ApertureSource(np.zeros((40,60,3),np.uint8),('x',),aperture)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.uint16,np.int64])
def test_source_range_never_guessed(dtype):
    with pytest.raises(ObservationError):ApertureSource(np.zeros((8,8,3),dtype),('x',))


def test_source_and_derived_arrays_own_immutable_storage():
    im=np.full((20,20,3),123,np.uint8);s=ApertureSource(im,('immutable',));im[:]=0
    assert s.image[0,0,0]==123
    g=MeasuredGroup(policy(),'group',(s,))
    for array in [s.image,g.features,g.aggregate,g.predicate_truth,g.rule_matches,*g.analyzed]:
        assert not array.flags.writeable
        with pytest.raises(ValueError):array.setflags(write=True)


def test_aperture_changes_features_but_never_pixels():
    im=np.zeros((40,60,3),np.uint8);im[8:32,12:48]=200
    before=im.copy();full=ApertureSource(im,('full',));crop=ApertureSource(im,('crop',),(12,8,48,32))
    assert measure_source(crop,policy())[0][2]==pytest.approx(200/255)
    assert measure_source(full,policy())[0][2]<200/255
    np.testing.assert_array_equal(im,before)


def test_outside_aperture_content_cannot_dominate_measurement():
    im=np.random.default_rng(3).integers(0,256,(40,60,3),dtype=np.uint8)
    a=ApertureSource(im,('a',),(12,8,48,32));im[:8]=0;im[32:]=255;im[:,:12]=0;im[:,48:]=255
    b=ApertureSource(im,('b',),(12,8,48,32))
    np.testing.assert_array_equal(measure_source(a,policy())[0],measure_source(b,policy())[0])
    assert a.image_sha256!=b.image_sha256


def test_whole_group_extrema_not_a_cherry_picked_source():
    p=demonstration_policy()
    a=ApertureSource(measured_record(39011,'clear')['image'],('a',),APERTURE)
    b=ApertureSource(measured_record(39012,'smooth')['image'],('b',),APERTURE)
    assert MeasuredGroup(p,'a',(a,)).category=='pixel_clear'
    mixed=MeasuredGroup(p,'g',(a,b))
    assert mixed.category=='pixel_smooth'
    np.testing.assert_array_equal(mixed.aggregate[0],mixed.features.min(0))
    np.testing.assert_array_equal(mixed.aggregate[1],mixed.features.max(0))
    assert mixed.lineage_ids==('a','b')


def test_group_order_does_not_change_receipt():
    a=source(100,lineage='a');b=source(101,lineage='b');p=policy()
    assert MeasuredGroup(p,'g',(a,b)).digest==MeasuredGroup(p,'g',(b,a)).digest


@pytest.mark.parametrize('members',[(),(source(),source()),(source(1),source(2),source(3))])
def test_empty_duplicate_and_overcapacity_group_rejected(members):
    with pytest.raises(ObservationError):MeasuredGroup(policy(),'g',members)


@pytest.mark.parametrize('op,threshold,expected',[('>=',40,True),('>',40,False),('<',40,False),('<=',40,True)])
def test_predicate_boundaries_are_explicit(op,threshold,expected):
    g=MeasuredGroup(policy(Predicate('min.native_short_edge_px',op,threshold)),'g',(source(),))
    assert bool(g.rule_matches[0])==expected
    assert (g.category=='inside')==expected


def test_overlap_has_no_first_rule_wins_behavior():
    rules=tuple(AcquisitionRule(n,(Predicate('min.native_short_edge_px','>=',4),)) for n in ('a','b'))
    for r in (rules,rules[::-1]):
        p=AcquisitionPolicy('overlap test',r);g=MeasuredGroup(p,'x',(source(),))
        assert g.category==AMBIGUOUS and not p.permits(g.category)
        assert g.rule_matches.all()


def test_no_match_is_distinct_from_zero_error():
    g=MeasuredGroup(policy(Predicate('min.native_short_edge_px','>',100)),'g',(source(),))
    assert g.category==UNMATCHED and not g.policy.permits(g.category)
    assert not g.rule_matches.any()


def test_explicit_matching_but_rejected_rule_remains_rejected():
    p=AcquisitionPolicy('no deployment',(AcquisitionRule('too_small',(Predicate('min.native_short_edge_px','>=',4),),False),))
    g=MeasuredGroup(p,'g',(source(),));assert g.category=='too_small' and not p.permits(g.category)


@pytest.mark.parametrize('value',[True,False,'0.1',None,np.nan,np.inf,-np.inf])
def test_predicates_reject_nonfinite_or_coerced_thresholds(value):
    with pytest.raises(ObservationError):Predicate('min.code_mean','>=',value)


@pytest.mark.parametrize('name,op',[('pair_similarity','>'),('min.detail_ratio','=='),('min.code_mean','eval'),('mean.code_mean','>')])
def test_only_fixed_source_metrics_and_comparators_allowed(name,op):
    with pytest.raises(ObservationError):Predicate(name,op,0.)


@pytest.mark.parametrize('kwargs',[dict(analysis_max_side=8),dict(analysis_max_side=True),dict(maximum_group_members=0),
    dict(maximum_group_members='2'),dict(low_code=True),dict(low_code=.9,high_code=.1),dict(variance_floor=0),dict(variance_floor=np.nan)])
def test_recipe_invalid_parameters(kwargs):
    with pytest.raises(ObservationError):replace(policy(),**kwargs)


def test_policy_serialization_preserves_all_thresholds_and_rejects_invented_fields():
    p=demonstration_policy();restored=AcquisitionPolicy.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.digest==p.digest
    assert policy_from_document(p.to_dict()).digest==p.digest
    for key,value in [('recipe','invented'),('feature_order',['bogus']),('radiometric_luminance',True)]:
        bad=p.to_dict();bad[key]=value
        with pytest.raises(ObservationError):AcquisitionPolicy.from_dict(bad)


def test_lineage_metadata_cannot_be_empty_duplicate_or_untrimmed():
    for ids in [(),('a','a'),('',),(' a',),('a ',)]:
        with pytest.raises(ObservationError):ApertureSource(np.zeros((8,8,3),np.uint8),ids)


def test_source_group_roundtrip_recomputes_original_pixels_and_all_arrays(tmp_path):
    g=MeasuredGroup(policy(),'g',(source(100,lineage='a'),source(200,lineage='b')))
    root=write_group(g,tmp_path/'group');again=load_group(root)
    assert again.digest==g.digest
    for key,a in g.arrays().items():np.testing.assert_array_equal(a,again.arrays()[key])
    with pytest.raises(FileExistsError):write_group(g,root)


def test_self_consistently_rehashed_feature_claims_are_recomputed(tmp_path):
    g=MeasuredGroup(policy(),'g',(source(),));root=write_group(g,tmp_path/'group')
    m=read_json(root/'measurements.json');m['features'][0][2]=.001
    (root/'measurements.json').write_text(json.dumps(m));rehash(root)
    with pytest.raises(ObservationError,match='recomputation'):load_group(root)


def test_self_consistently_rehashed_raw_arrays_are_recomputed(tmp_path):
    g=MeasuredGroup(policy(),'g',(source(),));root=write_group(g,tmp_path/'group')
    arrays=g.arrays();arrays['acquisition_features']=arrays['acquisition_features'].copy();arrays['acquisition_features'][0,0]=900
    np.savez_compressed(root/'arrays.npz',**arrays);rehash(root)
    with pytest.raises(ObservationError,match='arrays differ'):load_group(root)


def test_group_spec_cannot_supply_category_predictions_or_references(tmp_path):
    save_rgb(tmp_path/'im.png',source().image)
    spec={'format':'facemath.acquisition_group_spec.v9','group_id':'g','members':[{'image':'im.png','lineage_ids':['s']}]}
    path=tmp_path/'group.json';path.write_text(json.dumps(spec))
    assert group_from_spec(path,policy()).category=='inside'
    spec['category']='desired';path.write_text(json.dumps(spec))
    with pytest.raises(ObservationError):group_from_spec(path,policy())
    spec.pop('category');spec['members'][0]['predictions']=[[1,2]];path.write_text(json.dumps(spec))
    with pytest.raises(ObservationError):group_from_spec(path,policy())


def test_noise_is_an_explicit_counterexample_to_anatomical_quality_inference():
    noise=np.random.default_rng(39014).integers(30,220,(128,128,1),dtype=np.uint8);noise=np.repeat(noise,3,2)
    group=MeasuredGroup(demonstration_policy(),'noise',(ApertureSource(noise,('noise',)),))
    assert group.category=='pixel_clear'  # Literal high-detail code; not a face or valid localization.
    assert group.features[0,-1]>.30
