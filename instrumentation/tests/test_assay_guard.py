import copy, pathlib, unittest, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import assay_guard as ag

BASE = {
  "assay_id":"x","version":1,"scientific_question":"q",
  "null_source":{"name":"G0"},
  "alternatives":[{"name":"A"},{"name":"B"}],
  "target":{"name":"VMS","sealed_before_qualification":True},
  "sampling_unit":"trial","representation":{"name":"r"},
  "splits":{"development":1,"calibration":1,"validation":1,"controls":1,"target":"sealed"},
  "decision":{"metric":"T","max_null_rejection_rate":0.10,"min_alternative_rejection_rate":0.80,"min_effect_over_null_sd":2.0,"require_representation_robustness":True},
  "licensed_inference":"bounded","prohibited_inference":"global"
}

def summary(msha, a_rej=36, b_rej=34, effect=2.5):
    return {
      "manifest_sha256":msha,
      "null_validation":{"rejected":2,"n":50},
      "alternatives":[
        {"name":"A","rejected":a_rej,"n":40,"effect":effect,"null_sd":1.0},
        {"name":"B","rejected":b_rej,"n":40,"effect":effect,"null_sd":1.0}],
      "representation_checks":[{"name":"r2","pass":True}],
      "leakage_checks":{"target_inaccessible_during_fit":True,"disjoint_seed_namespaces":True,"training_only_model_selection":True}
    }

class TestGuard(unittest.TestCase):
    def freeze_rec(self, m):
        return {"manifest_sha256":ag.sha256_obj(m)}
    def test_pass(self):
        m=copy.deepcopy(BASE); s=summary(ag.sha256_obj(m)); q=ag.qualify(m,self.freeze_rec(m),s)
        self.assertTrue(q["overall_pass"]); ag.check_target(m,q)
    def test_power_fail_blocks(self):
        m=copy.deepcopy(BASE); s=summary(ag.sha256_obj(m), a_rej=10); q=ag.qualify(m,self.freeze_rec(m),s)
        self.assertFalse(q["overall_pass"])
        with self.assertRaises(PermissionError): ag.check_target(m,q)
    def test_under_2sd_blocks(self):
        m=copy.deepcopy(BASE); s=summary(ag.sha256_obj(m), effect=1.5); q=ag.qualify(m,self.freeze_rec(m),s)
        self.assertFalse(q["overall_pass"])
        self.assertEqual(q["alternatives"][0]["headline"],"THE METRIC DOES NOT RESOLVE THIS DEPARTURE")
    def test_manifest_mutation_breaks_target(self):
        m=copy.deepcopy(BASE); s=summary(ag.sha256_obj(m)); q=ag.qualify(m,self.freeze_rec(m),s)
        m["scientific_question"]="changed"
        with self.assertRaises(ValueError): ag.check_target(m,q)
    def test_missing_leakage_check_blocks(self):
        m=copy.deepcopy(BASE); s=summary(ag.sha256_obj(m)); del s["leakage_checks"]["training_only_model_selection"]
        q=ag.qualify(m,self.freeze_rec(m),s); self.assertFalse(q["overall_pass"])

if __name__=='__main__': unittest.main()
