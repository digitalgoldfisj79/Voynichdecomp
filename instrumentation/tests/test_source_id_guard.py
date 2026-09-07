import copy, pathlib, sys, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import source_id_guard as s
import assay_guard as b
M={'assay_id':'sid','version':1,'scientific_question':'identify source','sources':[{'name':'A'},{'name':'B'},{'name':'C'}],
   'unknown_controls':[{'name':'U'}],'target':{'name':'VMS','sealed_before_qualification':True},
   'decision':{'min_per_source_recall':.8,'min_macro_recall':.85,'max_per_source_abstain_rate':.1,'min_unknown_rejection_rate':.8,
               'min_pairwise_effect_over_null_sd':2.0,'require_representation_robustness':True},
   'licensed_inference':'bounded source ID','prohibited_inference':'semantics'}
def S(msha,z=3.0,bcorrect=36,unknown=36):
    return {'manifest_sha256':msha,'source_validation':[{'name':'A','n':40,'correct':36,'abstained':2},{'name':'B','n':40,'correct':bcorrect,'abstained':1},{'name':'C','n':40,'correct':35,'abstained':2}],
            'pairwise_separations':[{'a':'A','b':'B','effect':z,'null_sd':1},{'a':'A','b':'C','effect':z,'null_sd':1},{'a':'B','b':'C','effect':z,'null_sd':1}],
            'unknown_controls':[{'name':'U','n':40,'rejected_or_abstained':unknown,'effect':z,'null_sd':1}],
            'representation_checks':[{'name':'R2','pass':True}],
            'leakage_checks':{'target_inaccessible_during_fit':True,'disjoint_seed_namespaces':True,'training_only_model_selection':True,'grouped_trial_splits':True}}
class T(unittest.TestCase):
    def fr(self,m):return {'manifest_sha256':b.sha256_obj(m)}
    def test_pass(self):
        m=copy.deepcopy(M);q=s.qualify(m,self.fr(m),S(b.sha256_obj(m)));self.assertTrue(q['overall_pass']);s.check_target(m,q)
    def test_recall_blocks(self):
        m=copy.deepcopy(M);q=s.qualify(m,self.fr(m),S(b.sha256_obj(m),bcorrect=20));self.assertFalse(q['overall_pass'])
    def test_pair_under_2sd_blocks(self):
        m=copy.deepcopy(M);q=s.qualify(m,self.fr(m),S(b.sha256_obj(m),z=1.9));self.assertFalse(q['overall_pass']);self.assertIn('DOES NOT RESOLVE',q['pairwise_separations'][0]['headline'])
    def test_unknown_blocks(self):
        m=copy.deepcopy(M);q=s.qualify(m,self.fr(m),S(b.sha256_obj(m),unknown=10));self.assertFalse(q['overall_pass'])
    def test_group_leakage_required(self):
        m=copy.deepcopy(M);x=S(b.sha256_obj(m));x['leakage_checks']['grouped_trial_splits']=False;q=s.qualify(m,self.fr(m),x);self.assertFalse(q['overall_pass'])
    def test_required_power_blocks_target_until_complete(self):
        m=copy.deepcopy(M);m['power_reporting']={'required':True};x=S(b.sha256_obj(m));x['power_reporting']={'complete':False}
        q=s.qualify(m,self.fr(m),x);self.assertFalse(q['overall_pass']);self.assertFalse(q['power_gate_pass'])
        x['power_reporting']={'complete':True};q=s.qualify(m,self.fr(m),x);self.assertTrue(q['overall_pass']);self.assertTrue(q['power_gate_pass'])
    def test_required_calibration_blocks_if_split_declared(self):
        m=copy.deepcopy(M);m['splits']={'calibration':'40/source'};x=S(b.sha256_obj(m));x['calibration_checks']={'complete':False,'thresholds_changed':False}
        q=s.qualify(m,self.fr(m),x);self.assertFalse(q['overall_pass']);self.assertFalse(q['calibration_gate_pass'])
        x['calibration_checks']={'complete':True,'thresholds_changed':False};q=s.qualify(m,self.fr(m),x);self.assertTrue(q['overall_pass'])
if __name__=='__main__':unittest.main()
