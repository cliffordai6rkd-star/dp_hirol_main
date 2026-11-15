"""Shared, locked update accounting for the bounded CIVI engineering run."""
import csv
import fcntl
import json
from pathlib import Path

from diffusion_policy.distillation.dit_rgb import write_json


class UpdateBudget:
    def __init__(self, root, task_id, limit, run_dir):
        self.root, self.task, self.limit = Path(root), task_id, int(limit)
        self.run_dir = Path(run_dir)
        self.path = self.root / (task_id + '_budget.json')
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/(task_id+'.lock')).open('a+')
        with self:
            state = json.loads((self.root/'state.json').read_text())
            rows = list(csv.DictReader((self.root/'experiments.csv').open()))
            if sum(int(r['budget_counted']) for r in rows) != state['exploration_rgb_updates']:
                raise ValueError('Historical experiment ledger is inconsistent')
            if self.path.exists():
                ledger = json.loads(self.path.read_text())
                if ledger['limit'] != self.limit:
                    raise ValueError('A new explicit budget is required to change this task ceiling')
            else:
                if state.get('reserved_rgb_updates', 0):
                    raise ValueError('An existing reservation must finish before this run')
                if self.limit > state['exploration_budget']-state['exploration_rgb_updates']:
                    raise ValueError('Insufficient remaining exploration budget')
                ledger = dict(task=task_id, limit=self.limit, used=0, runs={})
                write_json(self.path, ledger)

    def __enter__(self):
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        fcntl.flock(self.lock, fcntl.LOCK_UN)

    def charge(self, step, samples):
        # Reserve and journal before optimizer.step. A crash can conservatively
        # count one unexecuted update; it can never silently exceed the ceiling.
        with self:
            ledger = json.loads(self.path.read_text())
            if ledger['used'] >= self.limit:
                raise RuntimeError('CIVI task optimizer update ceiling reached')
            state_path = self.root/'state.json'
            state = json.loads(state_path.read_text())
            if state['exploration_rgb_updates'] >= state['exploration_budget']:
                raise RuntimeError('Historical exploration ceiling reached')
            rows_path = self.root/'experiments.csv'
            with rows_path.open() as stream:
                reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames
            row = next((r for r in rows if r['id'] == self.task), None)
            if row is None:
                row = dict.fromkeys(fields, '')
                row.update(id=self.task, status='running', seed='42', teacher_step='150000',
                           rgb_updates='0', budget_counted='0', checkpoint=str(self.run_dir/'latest.pt'),
                           hypothesis='Formal direct-torque RGB frozen DiT implementation smoke',
                           promotion='False', reason='New objective, fresh renderer and optimizer; bounded engineering budget')
                rows.append(row)
            ledger['used'] += 1
            ledger['runs'][str(self.run_dir)] = ledger['runs'].get(str(self.run_dir), 0) + 1
            row['rgb_updates'] = row['budget_counted'] = str(int(row['budget_counted'])+1)
            state['exploration_rgb_updates'] += 1
            state['exploration_shared_rgb_updates'] = state.get('exploration_shared_rgb_updates', 0)+1
            temporary = rows_path.with_suffix('.csv.tmp')
            with temporary.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            temporary.replace(rows_path)
            write_json(state_path, state); write_json(self.path, ledger)
            with (self.run_dir/'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, task_update=ledger['used'], sample_indices=samples))+'\n')
