# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

added_files = [
    ( 'config.yaml', '.' ),
    ( 'train.yaml', '.' ),
    ( 'weights', 'weights' ),
    ( 'img_tmp', 'img_tmp' ),
    ( 'algorithms/RmFog/indoor_haze_best_3_6', 'algorithms/RmFog' ),
    ( 'algorithms/RmFog/outdoor_haze_best_3_6', 'algorithms/RmFog' ),
    ( 'algorithms/Yolov5/models/yolov5l.yaml', 'algorithms/Yolov5/models' ),
    ( 'algorithms/Yolov5/models/yolov5m.yaml', 'algorithms/Yolov5/models' ),
    ( 'algorithms/Yolov5/models/yolov5s.yaml', 'algorithms/Yolov5/models' ),
    ( 'algorithms/Yolov5/models/yolov5x.yaml', 'algorithms/Yolov5/models' ),
    ( 'algorithms/Yolov5/models/hub', 'algorithms/Yolov5/models/hub' ),
    ( 'algorithms/ZeroDCE/snapshots', 'algorithms/ZeroDCE/snapshots' ),
    ( 'inference', 'inference' )
]

a = Analysis(['main.py'],
             pathex=['algorithms', 'ui', '/home/dsm/codes/qt5_Detection'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
