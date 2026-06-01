# L2 / L3 collected metrics (seed 42; sweep 20260531)


## L2 — STL next-cat (next_gru), macro-F1 / accuracy

| engine | florida | alabama | arizona |
| --- | --- | --- | --- |
| hgi | 0.3429 / 0.3981 | 0.2591 / 0.3051 | 0.2821 / 0.3112 |
| check2hgi | 0.6732 / 0.7039 | 0.4704 / 0.5049 | 0.4695 / 0.5056 |
| check2hgi_design_b | 0.6458 / 0.6725 | 0.4266 / 0.4535 | 0.4500 / 0.4715 |
| check2hgi_resln | 0.6690 / 0.6996 | 0.4721 / 0.5007 | 0.4672 / 0.5048 |
| check2hgi_resln_design_b | 0.6714 / 0.7019 | 0.4727 / 0.5019 | 0.4751 / 0.5075 |

## L2 — capacity ladder (check2hgi: next_gru vs next_single), F1 / acc

| state | next_gru | next_single |
| --- | --- | --- |
| florida | 0.6732 / 0.7039 | 0.6555 / 0.6847 |
| alabama | 0.4704 / 0.5049 | 0.3835 / 0.4005 |
| arizona | 0.4695 / 0.5056 | 0.4292 / 0.4408 |

## L3 — MTL check2hgi_next_region (family only)

| engine | state | cat F1 | cat acc | reg acc | reg top5 | reg top10(indist) |
| --- | --- | --- | --- | --- | --- | --- |
| check2hgi | florida | 0.7028 | 0.7327 | 0.4717 | 0.5583 | 0.6025 |
| check2hgi | alabama | 0.4564 | 0.4720 | 0.2783 | 0.4268 | 0.5002 |
| check2hgi | arizona | 0.4807 | 0.4994 | 0.2665 | 0.3577 | 0.4117 |
| check2hgi_design_b | florida | 0.6873 | 0.7157 | 0.4712 | 0.5543 | 0.5980 |
| check2hgi_design_b | alabama | 0.4367 | 0.4483 | 0.2759 | 0.4201 | 0.4959 |
| check2hgi_design_b | arizona | 0.4587 | 0.4664 | 0.2708 | 0.3636 | 0.4161 |
| check2hgi_resln | florida | 0.7007 | 0.7308 | 0.4710 | 0.5548 | 0.5981 |
| check2hgi_resln | alabama | 0.4565 | 0.4791 | 0.2764 | 0.4185 | 0.4951 |
| check2hgi_resln | arizona | 0.4859 | 0.5129 | 0.2603 | 0.3424 | 0.3980 |
| check2hgi_resln_design_b | florida | 0.7020 | 0.7322 | 0.4714 | 0.5551 | 0.5985 |
| check2hgi_resln_design_b | alabama | 0.4668 | 0.4830 | 0.2770 | 0.4191 | 0.4947 |
| check2hgi_resln_design_b | arizona | 0.4816 | 0.5048 | 0.2667 | 0.3549 | 0.4087 |
