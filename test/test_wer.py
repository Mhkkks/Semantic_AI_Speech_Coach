from backend.wer_utils import compute_wer

ref = "I strongly support this policy"

hyp = "I strongly oppose this policy"

score = compute_wer(ref, hyp)

print(score)