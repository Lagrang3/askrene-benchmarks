import matplotlib.pyplot as plt
import numpy as np
import json

FEEPPM = 0.5 * 1e-2 * 1e6 # 0.5%
OVERTIME = 2000

def transform_scale(x):
    return np.log10(x)

def analize_runtime(data1, label1, data2, label2):
    '''
    Find the distribution of runtime as a function of the payment size.
    It includes all payments: success and fail, outliers (>OVERTIME) are
    excluded.
    '''
    def mkboxes(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        boxes = [[] for i in range(len(pos))]
        for x,y in zip(X,Y):
            boxes[idx[x]].append(y)
        return boxes, pos
    
    def analize_runtime_sample(data1, data2, sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample:
                    if d["runtime_msec"] > OVERTIME:
                            continue
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample
        
        def myplot(ax, data, label):
            x,y = filter_points(data)
            boxes, positions = mkboxes(transform_scale(x), y)
            x_ticks = [i for i in positions]
            x_labels = [("%d" % (10**i)) for i in x_ticks]
            ax.violinplot(boxes, positions = positions)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("amount (sat)")
            ax.set_title(label)
        
        fig = plt.figure(figsize=(9,4))
        ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)
        ax1.set_ylabel("runtime (ms)")
        fig.suptitle("Distribution of payment runtimes (%s nodes)" % sample)
        myplot(ax1, data1, label1)
        myplot(ax2, data2, label2)
        fig.savefig('runtime-%s.png' % sample)
    
    analize_runtime_sample(data1, data2, "big")
    analize_runtime_sample(data1, data2, "small")

def analize_overtime(data1, label1, data2, label2):
    '''
    Find the fraction of payments that take too long to compute as a function of the payment size.
    All payment outcomes are included: success and failure.
    '''
    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [ [0,0] for i in range(len(pos))]
        for x,y in zip(X,Y):
            freq[idx[x]][1] += 1
            if y > OVERTIME:
                freq[idx[x]][0] += 1
        return freq, pos
    
    def analize_overtime_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample
        fig = plt.figure()
        ax = fig.subplots()
        
        x1,y1=filter_points(data1)
        x_transf = transform_scale(x1)
        freq, positions = find_fraction(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.plot(positions, [ x/y for x,y in freq], '-o', label=label1)
        
        x2,y2=filter_points(data2)
        x_transf = transform_scale(x2)
        freq, positions = find_fraction(x_transf, y2)
        ax.plot(positions, [ x/y for x,y in freq], '-o', label=label2)
        
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("frequency")
        ax.set_title("Fraction of payments over %d msec (%s nodes)" % (OVERTIME, sample))
        fig.savefig('overtime-%s.png' % sample)
    
    analize_overtime_sample("big")
    analize_overtime_sample("small")


def analize_worsttime(data1, label1, data2, label2):
    '''
    Wort runtime.
    All payment outcomes are included: success and failure.
    '''
    def find_worst(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        worst = [ 0 for i in range(len(pos))]
        for x,y in zip(X,Y):
            current=worst[idx[x]]
            worst[idx[x]] = max(current, y)
        return worst, pos
    
    def analize_worsttime_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["runtime_msec"])
            return x_sample, y_sample
        fig = plt.figure()
        ax = fig.subplots()
        
        x1,y1=filter_points(data1)
        x_transf = transform_scale(x1)
        worst, positions = find_worst(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.plot(positions, worst, '-o', label=label1)
        
        x2,y2=filter_points(data2)
        x_transf = transform_scale(x2)
        worst, positions = find_worst(x_transf, y2)
        ax.plot(positions, worst, '-o', label=label2)
        
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("runtime (ms)")
        ax.set_title("Worst runtime (%s nodes)" % sample)
        fig.savefig('worsttime-%s.png' % sample)
    
    analize_worsttime_sample("big")
    analize_worsttime_sample("small")


def analize_failrate(data1, label1, data2, label2):
    '''
    Find the fraction of failed computations as a function of the payment size.
    '''
    def find_fraction(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        freq = [ [0,0] for i in range(len(pos))]
        for x,y in zip(X,Y):
            freq[idx[x]][1] += 1
            if y == False:
                freq[idx[x]][0] += 1
        return freq, pos
    
    def analize_failrate_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["success"])
            return x_sample, y_sample
        fig = plt.figure()
        ax = fig.subplots()
        
        x1,y1=filter_points(data1)
        x_transf = transform_scale(x1)
        freq1, positions = find_fraction(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        ax.plot(positions, [ x/y for x,y in freq1], '-o', label=label1)
        
        x2,y2=filter_points(data2)
        x_transf = transform_scale(x2)
        freq2, positions = find_fraction(x_transf, y2)
        ax.plot(positions, [ x/y for x,y in freq2], '-x', label=label2)
        
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("frequency")
        ax.set_title("Fraction of failed computations (%s nodes)" % sample)
        fig.savefig('failrate-%s.png' % sample)
    
    analize_failrate_sample("big")
    analize_failrate_sample("small")


def analize_fees(data1, label1, data2, label2):
    '''
    Find mean value of fees as a function of the payment size.
    It applies only to success payments.
    '''
    def find_mean(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        count = [ {"fee": 0, "count": 0} for i in range(len(pos))]
        for x,y in zip(X,Y):
            count[idx[x]]["count"] += 1
            count[idx[x]]["fee"] += y
        return count, pos
    
    def analize_fees_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample and d["success"]:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["fee_msat"])
            return x_sample, y_sample
        fig = plt.figure()
        ax = fig.subplots()
        
        x1,y1=filter_points(data1)
        x_transf = transform_scale(x1)
        count, positions = find_mean(x_transf, y1)
        x_ticks = [i for i in positions]
        x_labels = [("%d" % (10**i)) for i in x_ticks]
        fees = [ c["fee"]/c["count"] for c in count]
        feefrac = [ 1e3 * f/(10**t) for f,t in zip(fees, positions)]
        ax.plot(positions, feefrac, '-o', label=label1)
        
        x2,y2=filter_points(data2)
        x_transf = transform_scale(x2)
        count, positions = find_mean(x_transf, y2)
        x_ticks = [i for i in positions]
        fees = [ c["fee"]/c["count"] for c in count]
        feefrac = [ 1e3 * f/(10**t) for f,t in zip(fees, positions)]
        ax.plot(positions, feefrac, '-o', label=label2)
       
        ax.legend()
        ax.axhline(y=FEEPPM, color='r', linestyle='--')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("amount (sat)")
        ax.set_ylabel("mean fee (ppm)")
        ax.set_title("Fees per payment (%s nodes)" % sample)
        fig.savefig('fees-%s.png' % sample)
    
    analize_fees_sample("big")
    analize_fees_sample("small")


def analize_probability(data1, label1, data2, label2):
    '''
    Find the distribution of runtime as a function of the payment size.
    It includes all payments: success and fail, outliers (>OVERTIME) are
    excluded.
    '''
    def mkboxes(X, Y):
        pos = list(set([i for i in X]))
        pos.sort()
        idx = {}
        for i, x in enumerate(pos):
            idx[x] = i
        boxes = [[] for i in range(len(pos))]
        for x,y in zip(X,Y):
            boxes[idx[x]].append(-np.log(y))
        return boxes, pos
    
    def analize_probability_sample(sample):
        def filter_points(data):
            x_sample = []
            y_sample = []
            for d in data:
                if d["sample"]==sample and d["success"]:
                    x_sample.append(d["amount_msat"] * 1e-3)
                    y_sample.append(d["probability"])
            return x_sample, y_sample
        
        def myplot(ax, data, label):
            x,y = filter_points(data)
            boxes, positions = mkboxes(transform_scale(x), y)
            x_ticks = [i for i in positions]
            x_labels = [("%d" % (10**i)) for i in x_ticks]
            ax.violinplot(boxes, positions = positions)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("amount (sat)")
            ax.set_title(label)
        
        fig = plt.figure(figsize=(9,4))
        ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)
        ax1.set_ylabel("- log probability")
        fig.suptitle("Distribution of payment probabilities (%s nodes)" % sample)
        myplot(ax1, data1, label1)
        myplot(ax2, data2, label2)
        fig.savefig('probability-%s.png' % sample)
    
    analize_probability_sample("big")
    analize_probability_sample("small")


with open("prune-n-cap.json", "r") as fd:
    data_prune = json.load(fd)
with open("single-path.json", "r") as fd:
    data_default = json.load(fd)
with open("fix-constraints.json", "r") as fd:
    data_constraints = json.load(fd)

analize_runtime(data_default, "default", data_prune, "prune and cap")
analize_overtime(data_default, "default", data_prune, "prune and cap")
analize_worsttime(data_default, "default", data_prune, "prune and cap")
analize_failrate(data_default, "default", data_prune, "prune and cap")
analize_fees(data_default, "default", data_prune, "prune and cap")
analize_probability(data_default, "default", data_prune, "prune and cap")
