"""调度不依赖真实模型或数据；覆盖配额、队列耗尽及 FIFO 回退。"""
import unittest
from oeo_lab.experiments.common.runner import _prepare_pending, _next_pending_index


def item(name, budget=100, bins=5, identity='', seed=11, fold=1):
    return ({'model_name':name,'train_budget':budget,'max_bins':bins,'id':identity,'split':{'seed':seed,'fold':fold}}, None)


def active(rc, others):
    return [(None, item(name)[0], None, 0) for name in ['rc-oeo']*rc+['svm_rbf']*others]


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.settings={'scheduler':'babd_rc_oeo','rc_oeo_share':0.75,'n_jobs':30}

    def test_order_and_stable_ties(self):
        tasks=[item('svm_rbf',identity='s'),item('rc-oeo',100,50,'a'),
               item('rc-oeo',200,5,'b'),item('catboost',identity='c'),
               item('rc-oeo',200,50,'d'),item('rc-oeo',200,50,'e')]
        ordered=_prepare_pending(tasks,self.settings)
        self.assertEqual([t['id'] for t,_ in ordered],['d','b','a','e','s','c'])

    def test_seed_groups_and_combination_rotation(self):
        tasks=[item('rc-oeo',budget,bins,f'{seed}-{fold}-{budget}-{bins}',seed,fold)
               for seed in (37,11) for fold in range(1,6)
               for budget in (500,1000,30000) for bins in (5,20,50)]
        ordered=[t for t,_ in _prepare_pending(tasks,self.settings)]
        self.assertEqual([t['split']['seed'] for t in ordered], [37]*45+[11]*45)
        first=ordered[:9]
        self.assertEqual(len({(t['train_budget'],t['max_bins']) for t in first}),9)
        self.assertEqual((first[0]['train_budget'],first[0]['max_bins']),(30000,50))
        self.assertEqual((first[1]['train_budget'],first[1]['max_bins']),(500,5))
        self.assertTrue(all(t['split']['fold']==1 for t in first))
        largest=[t for t in ordered[:45] if t['train_budget']==30000 and t['max_bins']==50]
        self.assertEqual([t['split']['fold'] for t in largest],[1,2,3,4,5])
        self.assertEqual(len({t['id'] for t in ordered}),len(tasks))

    def test_resume_subset_no_missing_or_duplicate_tasks(self):
        tasks=[item('rc-oeo',b,m,f'{f}-{b}-{m}',11,f)
               for f in (1,2,3) for b in (500,30000) for m in (5,50)]
        pending=tasks[2:7]+tasks[9:]
        result=_prepare_pending(pending,self.settings)
        self.assertEqual(sorted(t['id'] for t,_ in result),sorted(t['id'] for t,_ in pending))
        self.assertEqual(result,_prepare_pending(pending,self.settings))

    def test_both_queues_reserve_slots(self):
        pending=[item('rc-oeo'),item('svm_rbf')]
        self.assertEqual(_next_pending_index(pending,active(21,8),self.settings),0)
        self.assertEqual(_next_pending_index(pending,active(22,7),self.settings),1)

    def test_share_as_soon_as_queue_empty(self):
        # 其他模型仍有 7 个运行中，但其队列已空，RC-OEO 可以超过原 22 个名额。
        self.assertEqual(_next_pending_index([item('rc-oeo')],active(22,7),self.settings),0)
        self.assertEqual(_next_pending_index([item('svm_rbf')],active(21,8),self.settings),0)

    def test_one_worker_and_one_group(self):
        self.settings['n_jobs']=1
        self.assertEqual(_next_pending_index([item('rc-oeo')],[],self.settings),0)
        self.assertEqual(_next_pending_index([item('svm_rbf')],[],self.settings),0)

    def test_fifo_unchanged_for_other_experiments(self):
        pending=[item('svm_rbf'),item('rc-oeo')]
        self.assertEqual(_prepare_pending(pending,{'n_jobs':30}),pending)
        self.assertEqual(_next_pending_index(pending,[],{'n_jobs':30}),0)

    def test_dispatch_simulation(self):
        for jobs in (1,2,3,4,20,30,38):
            self.settings['n_jobs']=jobs
            pending=_prepare_pending([item('svm_rbf',identity=f's{i}') for i in range(9)]+
                [item('rc-oeo',i+1,identity=f'r{i}') for i in range(70)],self.settings)
            running=[];seen=[]
            while pending or running:
                while pending and len(running)<jobs:
                    index=_next_pending_index(pending,running,self.settings)
                    self.assertIsNotNone(index)
                    task,_=pending.pop(index)
                    running.append((None,task,None,0));seen.append(task['id'])
                    self.assertLessEqual(len(running),jobs)
                if running:running.pop(0)
            self.assertEqual(len(seen),79)
            self.assertEqual(len(set(seen)),79)


if __name__=='__main__':unittest.main()
