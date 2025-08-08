from fixtures import *  # noqa: F401,F403
from hashlib import sha256
from pyln.client import RpcError
from pyln.testing.utils import SLOW_MACHINE
from utils import (
    only_one,
    first_scid,
    GenChannel,
    generate_gossip_store,
    sync_blockheight,
    wait_for,
    TEST_NETWORK,
    TIMEOUT,
)
import copy
import os
import pytest
import subprocess
import time
import tempfile
import random
import json


# @pytest.mark.slow_test
# def test_single(node_factory):
#     current_branch = subprocess.run(["git", "branch", "--show-current"],
#         check=True, capture_output=True).stdout.decode().split()[0]
#     outfile = tempfile.NamedTemporaryFile(prefix='gossip-store-')
#     # subprocess.check_output(['devtools/gossmap-compress',
#     #                          'decompress',
#     #                          'gossip_store.compressed',
#     #                          outfile.name])
#     subprocess.check_output(['rm',
#                              '-f',
#                              outfile.name])
#     subprocess.check_output(['ln',
#                              '-s',
#                              '/repo/gossip_store',
#                              outfile.name])
#
#
#     # This is in msat, but is also the size of channel we create.
#     AMOUNT = 100000000
#
#     l1 = node_factory.line_graph(1, fundamount=AMOUNT,
#                                      opts=[{'gossip_store_file': outfile.name,
#                                             'allow_warning': True,
#                                             'dev-throttle-gossip': None}])[0]
#
#     src = '0240b2b1f91d742327ebd34ea1129985ec98f9c08f7ab9a884a77ff3472897242e'
#     dst = '02fbd735ffb1c0ccab0ed05f5b4977130db0eb9154e777bee3fe588a9876e8080e'
#     amt = 10000000
#     MAX_FEE = max(amt // 200, 5000)
#     resp = l1.rpc.getroutes(source=src,
#                             destination=dst,
#                             amount_msat=amt,
#                             layers=[],
#                             maxfee_msat=MAX_FEE,
#                             final_cltv=18)
#     print(resp)


@pytest.mark.slow_test
def test_real_data(node_factory):
    current_branch = (
        subprocess.run(
            ["git", "branch", "--show-current"], check=True, capture_output=True
        )
        .stdout.decode()
        .split()[0]
    )
    outfile = tempfile.NamedTemporaryFile(prefix="gossip-store-")
    # subprocess.check_output(['devtools/gossmap-compress',
    #                          'decompress',
    #                          'gossip_store.compressed',
    #                          outfile.name])
    subprocess.check_output(["rm", "-f", outfile.name])
    subprocess.check_output(["ln", "-s", "/repo/gossip_store", outfile.name])

    # This is in msat, but is also the size of channel we create.
    AMOUNT = 100000000

    l1 = node_factory.line_graph(
        1,
        fundamount=AMOUNT,
        opts=[
            {
                "gossip_store_file": outfile.name,
                "allow_warning": True,
                "dev-throttle-gossip": None,
            }
        ],
    )[0]

    # node selection
    all_nodes = []
    all_chans = l1.rpc.listchannels()["channels"]

    node_dict = {}
    for c in all_chans:
        n = c["source"]
        if n not in node_dict:
            node_dict[n] = {"id": n, "num_chans": 0, "capacity": 0}
        node_dict[n]["num_chans"] += 1
        node_dict[n]["capacity"] += c["amount_msat"]

    all_nodes = [data for n, data in node_dict.items()]

    all_nodes.sort(key=lambda n: n["capacity"])

    N = len(all_nodes)
    all_small = all_nodes[int(0.10 * N) : int(0.35 * N)]
    all_big = all_nodes[int(0.65 * N) :]

    random.seed(42)
    amounts = [100, 1000, 10000, 100000, 1000000]
    num_samples = 10000
    datapoints = []

    def routes_fee(routes):
        pay = 0
        deliver = 0
        for r in routes["routes"]:
            deliver += r["amount_msat"]
            pay += r["path"][0]["amount_msat"]
        return pay - deliver

    def run_sim(node_set, amt_msat, repeat, version, sample_name, data):
        working_set = [
            copy.deepcopy(s) for s in node_set if s["capacity"] * 0.9 >= amt_msat
        ]
        for rep in range(repeat):
            # print(f"BENCHMARK: running repetition {rep}, sample {sample_name} and amount {amt_msat}")
            # 0.5% or 5sat is the norm
            MAX_FEE = max(amt_msat // 200, 5000)
            src = {}
            dst = {}
            if len(working_set) <= 2:
                break
            # print("BENCHMARK: Selecting source node")
            while True:
                src_index = random.randint(0, len(working_set) - 1)
                src = working_set[src_index]
                break
            # print("BENCHMARK: Selecting destination node")
            while True:
                dst_index = random.randint(0, len(working_set) - 1)
                if dst_index == src_index:
                    continue
                dst = working_set[dst_index]
                break
            try:
                # print("BENCHMARK: calling getroutes source=%s dest=%s" %
                #    (src["id"], dst["id"]))
                resp = l1.rpc.getroutes(
                    source=src["id"],
                    destination=dst["id"],
                    amount_msat=amt_msat,
                    layers=[],
                    maxfee_msat=MAX_FEE,
                    final_cltv=18,
                )
                success = True
            except RpcError as e:
                success = False
                resp = e.error
            # print(f"BENCHMARK: getroutes success {success}")
            timeline = l1.daemon.wait_for_log(
                "plugin-cln-askrene.*notify msg.*get_routes (completed|failed)"
            )
            err = ""
            if success == False:
                pattern = "getroutes failed reason:"
                errline = l1.daemon.wait_for_log(pattern)
                err = errline.split(pattern)[-1].strip()
            runtime = int(timeline.split()[-2])
            this_data = {
                "runtime_msec": runtime,
                "amount_msat": amt_msat,
                "version": version,
                "sample": sample_name,
                "success": success,
                "source": src["id"],
                "destination": dst["id"],
            }
            if success:
                this_data["probability"] = resp["probability_ppm"] * 1e-6
                this_data["fee_msat"] = routes_fee(resp)
            else:
                this_data["probability"] = 0.0
                this_data["fee_msat"] = 0
            this_data["fail_reason"] = err
            data.append(this_data)

    for amt_sat in amounts:
        run_sim(all_big, amt_sat * 1000, num_samples, current_branch, "big", datapoints)
        run_sim(
            all_small, amt_sat * 1000, num_samples, current_branch, "small", datapoints
        )
    with open(current_branch + ".json", "w") as fd:
        json.dump(datapoints, fd)
