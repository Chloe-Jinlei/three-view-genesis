"""Run the verified native IPC pipeline, then render its recorded states."""
import argparse,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='output/new_native_run')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--settle-steps',type=int,default=200)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--width',type=int,default=960)
    args=p.parse_args()
    out=(ROOT/args.out).resolve()
    subprocess.run(['bash',str(ROOT/'run_ipc.sh'),'--mode','dressing',
        '--steps',str(args.steps),'--settle-steps',str(args.settle_steps),
        '--dt',str(args.dt),'--no-render','--no-robots','--out',str(out)],check=True)
    report=json.loads((out/'report.json').read_text())
    if report.get('run_error') or report.get('completed_steps')!=args.steps+args.settle_steps:
        raise RuntimeError(f'Physical run did not finish; inspect {out / "report.json"}')
    subprocess.run([sys.executable,str(ROOT/'finite_arm_validate.py'),str(out)],check=True)
    subprocess.run([sys.executable,str(ROOT/'render_ipc_replay.py'),str(out/'trajectory.npz'),
        '--width',str(args.width)],check=True)
    # Success of the pipeline only means files were generated. Physical task
    # criteria and the limits of assisted actuation remain in the reports.
    print(f'Finished. Review the movie and validation in: {out}')

if __name__=='__main__':main()
