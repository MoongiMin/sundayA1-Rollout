"""Read-only manual leg range recorder. Never commands torque, goals or calibration."""
import argparse
import json
import math
from pathlib import Path
import signal
import select
import sys
import time

from rollout import LegBus

JOINTS = {j['id']: name for name, j in json.loads(Path(__file__).with_name('r25_contract.json').read_text(encoding='utf-8'))['joints'].items() if j['board']=='lower_body'}


def summarize(rows, selected):
    result={}
    for mid,name in JOINTS.items():
        values=[r['ticks'][str(mid)] for r in rows]
        if not values: continue
        changes=[v-values[0] for v in values if abs(v-values[0])>=8]
        result[str(mid)]={'joint':name,'first_tick':values[0], 'last_tick':values[-1],
                         'min_tick':min(values),'max_tick':max(values),
                         'span_ticks':max(values)-min(values),
                         'first_clear_change':'increase' if changes and changes[0]>0 else
                                              'decrease' if changes else 'not_detected',
                         'possible_wrap':any(abs(b-a)>2048 for a,b in zip(values,values[1:]))}
    return {'sample_count':len(rows),
            'duration_s':rows[-1]['elapsed_s']-rows[0]['elapsed_s'] if len(rows)>1 else 0,
            'selected_motor_id':selected,'joints':result,
            'moving_ids':[int(k) for k,v in result.items() if v['span_ticks']>=8],
            'insufficient_motion_ids':[int(k) for k,v in result.items() if v['span_ticks']<8],
            'other_moving_ids':[int(k) for k,v in result.items() if selected is not None and int(k)!=selected and v['span_ticks']>=8],
            'physical_limits_verified':False,'cad_zero_verified':False,
            'note':'Observed manual range only; no tick-to-radian sign, zero, or safe mechanical limits inferred.'}


def capture(bus, selected, seconds, log, clock=time, output=print, stop_requested=None):
    rows=[]
    start=clock.monotonic();last_print=-1
    try:
        while clock.monotonic()-start<seconds:
            if stop_requested is not None and stop_requested():break
            begun=clock.monotonic()
            raw=bus.positions()
            if set(raw)!=set(JOINTS) or any(type(v) is not int or not 0<=v<=4095 for v in raw.values()):
                raise ValueError('Invalid encoder sample / unsupported multi-turn position')
            # Catch an externally enabled target servo as well; never disable it here.
            for mid in JOINTS if selected is None else [selected]:
                if bus.read(mid,40,1)!=0:
                    raise RuntimeError(f'ID {mid} torque became enabled; read-only capture stopped')
            elapsed=clock.monotonic()-start
            row={'elapsed_s':elapsed,'monotonic_ns':clock.monotonic_ns(),
                 'ticks':{str(k):v for k,v in raw.items()},'motor_commands_sent':False}
            log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
            if len(rows)>1 and any(abs(row['ticks'][k]-rows[-2]['ticks'][k])>2048 for k in row['ticks']):
                raise ValueError('Possible encoder wrap; do not use this range as calibration')
            second=int(elapsed)
            if second!=last_print:
                ids=JOINTS if selected is None else [selected]
                lines=[f'{elapsed:5.1f}s — manual range capture (Enter to finish)',
                       ' ID  JOINT             MIN   NOW   MAX  SPAN']
                for mid in ids:
                    values=[r['ticks'][str(mid)] for r in rows]
                    lines.append(f'{mid:3d}  {JOINTS[mid]:16s} {min(values):5d} {raw[mid]:5d} {max(values):5d} {max(values)-min(values):5d}')
                output('\n'.join(lines))
                last_print=second
            if clock.monotonic()-begun>.5: raise TimeoutError('Read cycle exceeded 0.5 seconds')
            clock.sleep(max(0,.1-(clock.monotonic()-begun)))
    except BaseException as exc:
        return rows, 'interrupted' if isinstance(exc,KeyboardInterrupt) else f'{type(exc).__name__}: {exc}'
    return rows,None


def lerobot_candidate(summary, metadata, error):
    """Match MotorCalibration fields; never import/write automatically."""
    if error or summary['sample_count']<2 or summary['insufficient_motion_ids']:
        return None
    if set(summary['joints'])!={str(i) for i in JOINTS}:return None
    if any(v['possible_wrap'] for v in summary['joints'].values()):return None
    return {name:{'id':mid,'drive_mode':0,'homing_offset':metadata[str(mid)]['homing_offset'],
                  'range_min':summary['joints'][str(mid)]['min_tick'],
                  'range_max':summary['joints'][str(mid)]['max_tick']}
            for mid,name in JOINTS.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--id',type=int,choices=sorted(JOINTS))
    group.add_argument('--all',action='store_true',help='Record and summarize all 12 legs together')
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--delay',type=int,default=10)
    parser.add_argument('--first-motion',default='unspecified',help='Human description, e.g. bend')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.range_capture.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('manual_calibration'))
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or not 5<=args.seconds<=600:parser.error('seconds: 5..600')
    if not 0<=args.delay<=60:parser.error('delay: 0..60')
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    port=cfg['ports']['lower_body']
    if not port:raise ValueError('Configure ports.lower_body first')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stem=args.output_dir/f'{"all_legs" if args.all else "id"+str(args.id)}_{time.time_ns()}'
    bus=None;rows=[];error=None;metadata={}
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        bus=LegBus(port,cfg['baudrate'],list(JOINTS))
        for mid in JOINTS:
            model=bus.read(mid,3);mode=bus.read(mid,33,1);torque=bus.read(mid,40,1)
            raw=bus.read(mid,31);offset=-(raw&2047) if raw&2048 else raw
            metadata[str(mid)]={'model':model,'mode':mode,'torque_enabled':torque,'homing_offset':offset}
            if model!=777 or mode!=0:raise ValueError(f'ID {mid}: unexpected model/mode')
            if torque!=0:raise ValueError(f'ID {mid}: torque is ON; recorder will not change it')
        print('READ ONLY: no motor writes. Support the lying robot. Do not force hard stops.',flush=True)
        for n in range(args.delay,0,-1):
            print(f'Recording starts in {n}s',flush=True);time.sleep(1)
        print('RECORDING: move all leg joints by hand; Enter finishes.' if args.all else
              'RECORDING: move selected joint slowly; Enter finishes.',flush=True)
        print('Pause at each comfortable endpoint. No need to force hard stops.',flush=True)
        def enter_pressed():
            if not sys.stdin.isatty():return False
            if select.select([sys.stdin],[],[],0)[0]:
                sys.stdin.readline()
                return True
            return False
        display_lines=0
        def display(table):
            nonlocal display_lines
            if sys.stdout.isatty() and display_lines:
                sys.stdout.write(f'\033[{display_lines}A')
            print(table,flush=True)
            display_lines=len(table.splitlines())
        with stem.with_suffix('.raw.jsonl').open('x',encoding='utf-8') as log:
            rows,error=capture(bus,args.id,args.seconds,log,output=display,stop_requested=enter_pressed)
    except KeyboardInterrupt:error='interrupted before recording'
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if bus is not None:
            try:bus.close()
            except Exception as exc:error=(error or '')+f'; close failed: {exc}'
    result=summarize(rows,args.id)
    result.update(port=port,first_motion_description=args.first_motion,motor_settings=metadata,
                  completed=error is None,error=error,motor_commands_sent=False,
                  mode='all_legs' if args.all else 'single_joint')
    with stem.with_suffix('.summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('Summary file:',stem.with_suffix('.summary.json'),flush=True)
    if str(args.id) in result['joints']:
        print(json.dumps(result['joints'][str(args.id)],indent=2),flush=True)
        print('Other moving IDs:',result['other_moving_ids'],flush=True)
    elif args.all:
        print(' ID  JOINT             MIN   MAX  SPAN')
        for mid,name in JOINTS.items():
            item=result['joints'].get(str(mid))
            if item:print(f'{mid:3d}  {name:16s} {item["min_tick"]:5d} {item["max_tick"]:5d} {item["span_ticks"]:5d}')
        print('Insufficient motion (<8 ticks):',result['insufficient_motion_ids'])
        candidate=lerobot_candidate(result,metadata,error)
        if candidate is not None:
            candidate_path=stem.with_suffix('.lerobot_candidate.json')
            with candidate_path.open('x',encoding='utf-8') as f:json.dump(candidate,f,indent=2)
            print('LeRobot-format range candidate (not applied):',candidate_path)
        else:print('No complete LeRobot candidate exported; review summary/raw data.')
    if error:print('Capture ended:',error,flush=True)
    print('No calibration or motor settings were changed.',flush=True)
    return 0 if error is None else 2


if __name__=='__main__':raise SystemExit(main())
