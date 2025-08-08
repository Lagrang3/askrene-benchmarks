import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import re

FEEPPM = 0.5 * 1e-2 * 1e6  # 0.5%
OVERTIME = 200


def transform_scale(x):
    return np.log10(x)


def analize_runtime(data1, label1, data2, label2):
    """
    Find the distribution of runtime as a function of the payment size.
    It includes all payments: success and fail, outliers (>OVERTIME) are
    excluded.
    """

    def mkboxes(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        boxes = [[] for i in range(len(pos))]
        for x, y in zip(X, Y):
            boxes[idx[x]].append(y)
        return boxes, pos

    def analize_runtime_sample(data1, data2, sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample:
                    if d["runtime_msec"] > OVERTIME:
                        continue
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample

        def myplot(ax, data, label):
            x, y = filter_points(data)
            boxes, positions = mkboxes(transform_scale(x), y)
            x_ticks = [i for i in positions]
            x_labels = [("%d" % (10**i)) for i in x_ticks]
            ax.violinplot(boxes, positions=positions)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("amount (sat)")
            ax.set_title(label)
            return [np.mean(b) for b in boxes]

        fig = plt.figure(figsize=(9, 4))
        ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)
        ax1.set_ylabel("runtime (ms)")
        fig.suptitle("Distribution of payment runtimes (%s nodes)" % sample)
        mean1 = myplot(ax1, data1, label1)
        mean2 = myplot(ax2, data2, label2)
        fig.savefig("runtime-%s.png" % sample)

    analize_runtime_sample(data1, data2, "big")
    # analize_runtime_sample(data1, data2, "small")


def analize_overtime(data1, label1, data2, label2):
    """
    Find the fraction of payments that take too long to compute as a function of the payment size.
    All payment outcomes are included: success and failure.
    """

    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [[0, 0] for i in range(len(pos))]
        for x, y in zip(X, Y):
            freq[idx[x]][1] += 1
            if y > OVERTIME:
                freq[idx[x]][0] += 1
        return freq, pos

    def analize_overtime_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample

        fig = plt.figure()
        ax = fig.subplots()

        x1, y1 = filter_points(data1)
        x_transf = transform_scale(x1)
        freq, positions = find_fraction(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ratio = [x / y for x, y in freq]
        print(ratio)
        ax.plot(positions, ratio, "-o", label=label1)

        x2, y2 = filter_points(data2)
        x_transf = transform_scale(x2)
        freq, positions = find_fraction(x_transf, y2)
        ratio = [x / y for x, y in freq]
        print(ratio)
        ax.plot(positions, ratio, "-o", label=label2)

        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("frequency")
        ax.set_title(
            "Fraction of payments over %d msec (%s nodes)" % (OVERTIME, sample)
        )
        fig.savefig("overtime-%s.png" % sample)

    analize_overtime_sample("big")
    # analize_overtime_sample("small")


def analize_worsttime(data1, label1, data2, label2):
    """
    Wort runtime.
    All payment outcomes are included: success and failure.
    """

    def find_worst(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        worst = [0 for i in range(len(pos))]
        for x, y in zip(X, Y):
            current = worst[idx[x]]
            worst[idx[x]] = max(current, y)
        return worst, pos

    def analize_worsttime_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample

        fig = plt.figure()
        ax = fig.subplots()

        x1, y1 = filter_points(data1)
        x_transf = transform_scale(x1)
        worst, positions = find_worst(x_transf, y1)
        print(worst)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.semilogy(positions, worst, "-o", label=label1)

        x2, y2 = filter_points(data2)
        x_transf = transform_scale(x2)
        worst, positions = find_worst(x_transf, y2)
        print(worst)
        ax.semilogy(positions, worst, "-o", label=label2)

        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("runtime (ms)")
        ax.set_title("Worst runtime (%s nodes)" % sample)
        fig.savefig("worsttime-%s.png" % sample)

    analize_worsttime_sample("big")
    # analize_worsttime_sample("small")


def analize_meantime(data1, label1, data2, label2):
    """
    Mean runtime.
    All payment outcomes are included: success and failure.
    """

    def find_mean(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        all_values = [[] for i in range(len(pos))]
        for x, y in zip(X, Y):
            all_values[idx[x]].append(y)
        mean = [np.mean(values) for values in all_values]
        return mean, pos

    def analize_meantime_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample

        fig = plt.figure()
        ax = fig.subplots()

        x1, y1 = filter_points(data1)
        x_transf = transform_scale(x1)
        mean, positions = find_mean(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.plot(positions, mean, "-o", label=label1)
        time_vals1 = [float(x) for x in mean]
        print(positions, time_vals1, label1)

        x2, y2 = filter_points(data2)
        x_transf = transform_scale(x2)
        mean, positions = find_mean(x_transf, y2)
        time_vals2 = [float(x) for x in mean]
        ax.plot(positions, mean, "-o", label=label2)
        print(positions, time_vals2, label1)

        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("runtime (ms)")
        ax.set_title("Mean runtime (%s nodes)" % sample)
        fig.savefig("meantime-%s.png" % sample)

        print([x / y for x, y in zip(time_vals1, time_vals2)])

    analize_meantime_sample("big")
    # analize_meantime_sample("small")


def analize_failrate(data1, label1, data2, label2):
    """
    Find the fraction of failed computations as a function of the payment size.
    """

    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [[0, 0] for i in range(len(pos))]
        for x, y in zip(X, Y):
            freq[idx[x]][1] += 1
            if y == False:
                freq[idx[x]][0] += 1
        return freq, pos

    def analize_failrate_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["success"])
            return x_sample, y_sample

        fig = plt.figure()
        ax = fig.subplots()

        x1, y1 = filter_points(data1)
        x_transf = transform_scale(x1)
        freq1, positions = find_fraction(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.plot(positions, [x / y for x, y in freq1], "-o", label=label1)

        x2, y2 = filter_points(data2)
        x_transf = transform_scale(x2)
        freq2, positions = find_fraction(x_transf, y2)
        ax.plot(positions, [x / y for x, y in freq2], "-x", label=label2)

        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("frequency")
        ax.set_title("Fraction of failed computations (%s nodes)" % sample)
        fig.savefig("failrate-%s.png" % sample)

    analize_failrate_sample("big")
    analize_failrate_sample("small")


def analize_fees(data1, label1, data2, label2):
    """
    Find mean value of fees as a function of the payment size.
    It applies only to success payments.
    """

    def find_mean(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        count = [{"fee": 0, "count": 0} for i in range(len(pos))]
        for x, y in zip(X, Y):
            count[idx[x]]["count"] += 1
            count[idx[x]]["fee"] += y
        return count, pos

    def analize_fees_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample and d["success"]:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["fee_msat"])
            return x_sample, y_sample

        fig = plt.figure()
        ax = fig.subplots()

        x1, y1 = filter_points(data1)
        x_transf = transform_scale(x1)
        count, positions = find_mean(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        fees = [c["fee"] / c["count"] for c in count]
        feefrac = [1e3 * f / (10**t) for f, t in zip(fees, positions)]
        ax.plot(positions, feefrac, "-o", label=label1)

        x2, y2 = filter_points(data2)
        x_transf = transform_scale(x2)
        count, positions = find_mean(x_transf, y2)
        x_ticks = [i for i in positions]
        fees = [c["fee"] / c["count"] for c in count]
        feefrac = [1e3 * f / (10**t) for f, t in zip(fees, positions)]
        ax.plot(positions, feefrac, "-o", label=label2)

        ax.legend()
        ax.axhline(y=FEEPPM, color="r", linestyle="--")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("mean fee (ppm)")
        ax.set_title("Fees per payment (%s nodes)" % sample)
        fig.savefig("fees-%s.png" % sample)

    analize_fees_sample("big")
    analize_fees_sample("small")


def analize_probability(data1, label1, data2, label2):
    """
    Find the distribution of runtime as a function of the payment size.
    It includes all payments: success and fail, outliers (>OVERTIME) are
    excluded.
    """

    def mkboxes(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        boxes = [[] for i in range(len(pos))]
        for x, y in zip(X, Y):
            boxes[idx[x]].append(-np.log(y))
        return boxes, pos

    def analize_probability_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"] == sample and d["success"]:
                    if d["probability"] < 1e-1:
                        continue
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["probability"])
            return x_sample, y_sample

        def myplot(ax, data, label):
            x, y = filter_points(data)
            boxes, positions = mkboxes(transform_scale(x), y)
            x_ticks = [i for i in positions]
            x_labels = [("%d" % (10**i)) for i in x_ticks]
            ax.violinplot(boxes, positions=positions)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("amount (sat)")
            ax.set_title(label)

        fig = plt.figure(figsize=(9, 4))
        ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)
        ax1.set_ylabel("- log probability")
        fig.suptitle("Distribution of payment probabilities (%s nodes)" % sample)
        myplot(ax1, data1, label1)
        myplot(ax2, data2, label2)
        fig.savefig("probability-%s.png" % sample)

    analize_probability_sample("big")
    analize_probability_sample("small")


def compare_fail_rate(default_name, ref_name, sizes):
    def get_data(fname):
        with open(fname, "r") as fd:
            data = json.load(fd)
        return data

    def filter_points(data, sample):
        x_sample = []
        y_sample = []
        for d in data:
            if d["sample"] == sample:
                x_sample.append(d["amount_msat"] * 1e-3)
                y_sample.append(d["success"])
        return x_sample, y_sample

    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [[0, 0] for i in range(len(pos))]
        for x, y in zip(X, Y):
            freq[idx[x]][1] += 1
            if y == False:
                freq[idx[x]][0] += 1
        return freq, pos

    default_data = get_data(default_name + ".json")
    ref_data = [get_data("%d/%s.json" % (i, ref_name)) for i in sizes]
    sample = "big"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.subplots()

    for d, i in zip(ref_data, sizes):
        x, y = filter_points(d, sample)
        x_scaled = transform_scale(x)
        freq, pos = find_fraction(x_scaled, y)
        label = "precision=%d" % i
        ax.plot(pos, [x / y for x, y in freq], "-o", label=label)

    x, y = filter_points(default_data, sample)
    x_scaled = transform_scale(x)
    freq, pos = find_fraction(x_scaled, y)

    x_ticks = [i for i in pos]
    x_labels = [("%d" % (10**i)) for i in x_ticks]
    ax.plot(pos, [x / y for x, y in freq], "-o", label="master", color="black")

    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("amount (sat)")
    ax.set_ylabel("frequency")
    ax.set_title("Fraction of failed computations")
    fig.savefig("failrate.png")


def compare_feerate(default_name, ref_name, sizes):
    def get_data(fname):
        with open(fname, "r") as fd:
            data = json.load(fd)
        return data

    def find_mean(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        count = [{"fee": 0, "count": 0} for i in range(len(pos))]
        for x, y in zip(X, Y):
            count[idx[x]]["count"] += 1
            count[idx[x]]["fee"] += y
        return count, pos

    def filter_points(data, sample):
        x_sample = []
        y_sample = []
        for d in data:
            if d["sample"] == sample and d["success"]:
                x_sample.append(d["amount_msat"] * 1e-3)
                y_sample.append(d["fee_msat"])
        return x_sample, y_sample

    default_data = get_data(default_name + ".json")
    ref_data = [get_data("%d/%s.json" % (i, ref_name)) for i in sizes]
    sample = "big"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.subplots()

    for d, i in zip(ref_data, sizes):
        x, y = filter_points(d, sample)
        x_scaled = transform_scale(x)
        count, pos = find_mean(x_scaled, y)
        label = "precision=%d" % i
        fees = [c["fee"] / c["count"] for c in count]
        feefrac = [1e3 * f / (10**t) for f, t in zip(fees, pos)]
        ax.plot(pos, feefrac, "-o", label=label)

    x, y = filter_points(default_data, sample)
    x_scaled = transform_scale(x)
    count, pos = find_mean(x_scaled, y)

    x_ticks = [i for i in pos]
    x_labels = [("%d" % (10**i)) for i in x_ticks]
    fees = [c["fee"] / c["count"] for c in count]
    feefrac = [1e3 * f / (10**t) for f, t in zip(fees, pos)]
    ax.plot(pos, feefrac, "-o", label="master", color="black")

    ax.legend()
    ax.axhline(y=FEEPPM, color="r", linestyle="--")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("amount (sat)")
    ax.set_ylabel("mean fee (ppm)")
    ax.set_title("Fees per payment")
    fig.savefig("feerate.png")


def compare_worsttime(default_name, ref_name, sizes):
    def get_data(fname):
        with open(fname, "r") as fd:
            data = json.load(fd)
        return data

    def find_worst(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        worst = [0 for i in range(len(pos))]
        for x, y in zip(X, Y):
            current = worst[idx[x]]
            worst[idx[x]] = max(current, y)
        return worst, pos

    def filter_points(data, sample):
        x_sample = []
        y_sample = []
        for d in data:
            if d["sample"] == sample:
                x_sample.append(d["amount_msat"] * 1e-3)
                y_sample.append(d["runtime_msec"])
        return x_sample, y_sample

    default_data = get_data(default_name + ".json")
    ref_data = [get_data("%d/%s.json" % (i, ref_name)) for i in sizes]
    sample = "big"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.subplots()

    for d, i in zip(ref_data, sizes):
        x, y = filter_points(d, sample)
        x_scaled = transform_scale(x)
        worst, pos = find_worst(x_scaled, y)
        label = "precision=%d" % i
        ax.semilogy(pos, worst, "-o", label=label)

    x, y = filter_points(default_data, sample)
    x_scaled = transform_scale(x)
    worst, pos = find_worst(x_scaled, y)

    x_ticks = [i for i in pos]
    x_labels = [("%d" % (10**i)) for i in x_ticks]
    ax.semilogy(pos, worst, "-o", label="master", color="black")

    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("amount (sat)")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Worst runtime")
    fig.savefig("worsttime.png")


def compare_overtime(default_name, ref_name, sizes):
    def get_data(fname):
        with open(fname, "r") as fd:
            data = json.load(fd)
        return data

    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [[0, 0] for i in range(len(pos))]
        for x, y in zip(X, Y):
            freq[idx[x]][1] += 1
            if y > OVERTIME:
                freq[idx[x]][0] += 1
        return freq, pos

    def filter_points(data, sample):
        x_sample = []
        y_sample = []
        for d in data:
            if d["sample"] == sample:
                x_sample.append(d["amount_msat"] * 1e-3)
                y_sample.append(d["runtime_msec"])
        return x_sample, y_sample

    default_data = get_data(default_name + ".json")
    ref_data = [get_data("%d/%s.json" % (i, ref_name)) for i in sizes]
    sample = "big"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.subplots()

    for d, i in zip(ref_data, sizes):
        x, y = filter_points(d, sample)
        x_scaled = transform_scale(x)
        freq, pos = find_fraction(x_scaled, y)
        label = "precision=%d" % i
        ax.plot(pos, [x / y for x, y in freq], "-o", label=label)

    x, y = filter_points(default_data, sample)
    x_scaled = transform_scale(x)
    freq, pos = find_fraction(x_scaled, y)

    x_ticks = [i for i in pos]
    x_labels = [("%d" % (10**i)) for i in x_ticks]
    ax.plot(pos, [x / y for x, y in freq], "-o", label="master", color="black")

    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("amount (sat)")
    ax.set_ylabel("frequency")
    ax.set_title("Fraction of payments over %d msec" % OVERTIME)
    fig.savefig("overtime.png")


def analize_failure_one_to_one(data1, label1, data2, label2):
    sample = "big"

    reason_types = {
        "We couldn't quite afford it": 0,
        "Could not find route without excessive cost": 0,
        "Amount .* below minimum": 0,
        "Could not find route without excessive cost": 0,
        "check_htlc_min_limits failed": 0,
        "other": 0,
    }
    all_fail_types = {
        "We couldn't quite afford it": 0,
        "Could not find route without excessive cost": 0,
        "Amount .* below minimum": 0,
        "Could not find route without excessive cost": 0,
        "check_htlc_min_limits failed": 0,
        "The destination has disabled": 0,
        "The source has disabled": 0,
        "The shortest path is .*, but .* marked disabled by gossip message": 0,
        "Missing gossip for destination": 0,
        "The shortest path is .*, but .* exceeds htlc_maximum_msat": 0,
        "The shortest path is .*, but .* below htlc_minumum_msat": 0,
        "The shortest path is .*, but .* is constrained": 0,
        "The shortest path is .*, but .* has no gossip": 0,
        "The shortest path is .*, but .* isn't big enough to carry": 0,
        "other": 0,
    }

    ff = 0
    sf = 0
    fs = 0
    ss = 0
    bad_flips = []
    good_flips = []
    for d1, d2 in zip(data1, data2):
        assert d1["sample"] == d2["sample"]
        assert d1["amount_msat"] == d2["amount_msat"]
        assert d1["source"] == d2["source"]
        assert d1["destination"] == d2["destination"]

        if d1["sample"] != sample:
            continue
        if not d2["success"] and not d1["success"]:
            ff += 1
            nomatch = True
            for rt in all_fail_types:
                match = re.search(rt, d2["fail_reason"])
                if match:
                    all_fail_types[rt] += 1
                    nomatch = False
            if nomatch:
                print("unknown type", d2["fail_reason"])
                all_fail_types["other"] += 1
        if not d2["success"] and d1["success"]:
            fs += 1
            good_flips.append(copy.deepcopy(d2))
        if d2["success"] and d1["success"]:
            ss += 1
        if d2["success"] and not d1["success"]:
            sf += 1
            bad_flips.append(copy.deepcopy(d1))

    for d in good_flips:
        print(f"good flip:", d["fail_reason"])
        nomatch = True
        for rt in reason_types:
            match = re.search(rt, d["fail_reason"])
            if match:
                reason_types[rt] += 1
                nomatch = False
        if nomatch:
            reason_types["other"] += 1

    for d in bad_flips:
        print(f"bad flip: {d}")

    print(f"{ss} {fs}")
    print(f"{sf} {ff}")
    print("failed reasons for flips", reason_types)
    print("failed reasons for FF", all_fail_types)


with open("../askrene-prune-and-cap/askrene-prune-and-cap.json", "r") as fd:
    data_prune = json.load(fd)
with open("../askrene-fix-constraints/askrene-fix-constraints.json", "r") as fd:
    data_constraints = json.load(fd)

analize_overtime(data_constraints, "PR 8358", data_prune, "master")
analize_meantime(data_constraints, "PR 8358", data_prune, "master")
analize_runtime(data_constraints, "PR 8358", data_prune, "master")
analize_worsttime(data_constraints, "PR 8358", data_prune, "master")
analize_failrate(data_constraints, "PR 8358", data_prune, "master")
analize_fees(data_constraints, "PR 8358", data_prune, "master")
analize_probability(data_constraints, "PR 8358", data_prune, "master")


analize_failure_one_to_one(data_constraints, "PR 8358", data_prune, "master")


# with open("../askrene-prune-and-cap/compressed.json", "r") as fd:
#     data_compressed = json.load(fd)
# with open("../askrene-prune-and-cap/normal.json", "r") as fd:
#     data_normal = json.load(fd)
# analize_failrate(data_compressed, "compressed gossip_store", data_normal, "plain gossip_store")

# sizes = [1, 10, 100, 1000, 1000000]
# ref_name = "askrene-prune-and-cap"
# default_name = "askrene-single-path-solver"
#
# compare_fail_rate(default_name, ref_name, sizes)
# compare_feerate(default_name, ref_name, sizes)
# compare_worsttime(default_name, ref_name, sizes)
# compare_overtime(default_name, ref_name, sizes)
