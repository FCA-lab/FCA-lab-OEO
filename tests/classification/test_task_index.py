import copy
import json
import tempfile
import unittest
from pathlib import Path
from oeo_lab.experiments.common.task_index import save_task_index, iter_task_index
from oeo_lab.experiments.common.checkpoints import task_id, same_task, completed, save

class TaskIndexTests(unittest.TestCase):
    def test_shared_roundtrip_and_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'index.json'
            task={'dataset':{'path':'example.csv'}, 'model':{'input':'context'},
                  'provenance':{'source_hash':'old'}, 'max_bins':5,
                  'split':{'seed':11,'fold':1,'train_indices':list(range(500)),
                           'test_indices':list(range(500,10000))}}
            task['id']=task_id(task)
            other=copy.deepcopy(task);other['max_bins']=20;other['id']=task_id(other)
            original=copy.deepcopy([task,other])
            old=Path(d)/'old.json';save(old,{'tasks':original})
            self.assertEqual(list(iter_task_index(old)),original)
            save_task_index(p,iter_task_index(old))
            restored=list(iter_task_index(p))
            self.assertEqual(restored,original)
            self.assertEqual([task,other],original)
            self.assertNotIn('train_indices',p.read_text())
            self.assertLess(p.stat().st_size,old.stat().st_size/10)
            # 三份共享配置、两份数组、一份划分；第二个任务不再重复存储。
            self.assertEqual(len(list((Path(d)/'index_data').glob('*.json'))),6)
            checkpoint=Path(d)/'checkpoint.json';save(checkpoint,{'task':task,'status':'ok'})
            self.assertTrue(completed(checkpoint,restored[0]))
            self.assertTrue(same_task(task,restored[0]))
            self.assertEqual(task_id(restored[0]),task['id'])
            changed=copy.deepcopy(restored[0]);changed['split']['test_indices'][0]=-1
            self.assertFalse(same_task(task,changed))

    def test_empty_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'index.json';save_task_index(p,[])
            self.assertEqual(list(iter_task_index(p)),[])
            save_task_index(p,[{'split':{'train_indices':[1],'test_indices':[2]}}])
            data=json.loads(p.read_text());key=data['tasks'][0]['_index_refs']['split']
            (Path(d)/'index_data'/f'{key}.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                list(iter_task_index(p))

if __name__=='__main__':unittest.main()
