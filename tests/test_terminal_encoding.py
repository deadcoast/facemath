import os
import subprocess
import sys

from facemath.observation.readout import terminal_readout


def test_terminal_fallback_retains_values_and_unrepresentable_path():
    data={'kind':'observer_study','enrollment':{'committed':3,'eligible':4,'prospectively_excluded':1},
          'report':'C:/portraits/\u9762/report.html'}
    unicode_text=terminal_readout(data)
    text=terminal_readout(data,encoding='cp1252')
    assert '3/4 committed; 1 excluded' in text
    assert r'C:/portraits/\u9762/report.html' in text
    assert text.encode('cp1252').decode('cp1252')==text
    assert '◆' in unicode_text and '\u9762' in unicode_text


def test_redirected_overview_succeeds_with_strict_windows_encoding(tmp_path):
    from facemath.observation.artifacts import write_bundle
    from facemath.observation.canonical import InstrumentConfig
    from facemath.observation.instrument import FacialInstrument
    from facemath.observation.lab import phantom_points, phantom_observation, render_phantom
    points=phantom_points();image=render_phantom(points);obs=phantom_observation(image,points)
    result=FacialInstrument(InstrumentConfig(canvas_size=96,eye_span_px=32,uncertainty_draws=4,enable_flow=False,enable_edges=False)).run(image,image,obs,obs)
    bundle=tmp_path/'pair';write_bundle(result,bundle,image,image)
    process=subprocess.run([sys.executable,'-m','facemath','overview',str(bundle)],
        cwd=tmp_path,env=dict(os.environ,PYTHONIOENCODING='cp1252:strict'),capture_output=True)
    assert process.returncode==0,process.stderr.decode('cp1252')
    output=process.stdout.decode('cp1252')
    assert 'Regional movement and deformation' in output
    assert 'Facial structure' in output
    assert str(bundle/'report.html') in output
