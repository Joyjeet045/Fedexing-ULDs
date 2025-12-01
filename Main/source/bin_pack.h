#include "packing_structures.h"
#include <bits/stdc++.h>
using namespace std;

#pragma once

const int inf = 32767;

struct layerlist {
    int layereval;
    int layerdim;
} layers[1000];

struct scrappad {
    scrappad *pre, *pos;
    int cumx, cumz;
};

int complayerlist(const void *i, const void *j) {
    return *(long int *)i - *(long int *)j;
}

class Binpack {
  public:
    string strpx, strpy, strpz;
    string strcox, strcoy, strcoz;
    string strpackx, strpacky, strpackz;

    string strtemp;
    int tbn;
    int packing;
    int layerdone;
    int evened;
    int bestvariant;
    int packingbest;
    int hundredpercent;
    int unpacked;

    int boxx, boxy, boxz, boxi;
    int bboxx, bboxy, bboxz, bboxi;
    int cboxx, cboxy, cboxz, cboxi;
    int bfx, bfy, bfz;
    int bbfx, bbfy, bbfz;
    int xx, yy, zz;
    int px, py, pz;

    int x;
    int n;
    int layerlistlen;
    int layerinlayer;
    int prelayer;
    int lilz;
    int itenum;
    int hour;
    int min;
    int sec;
    int layersindex;
    int remainpx, remainpy, remainpz;
    int packedy;
    int prepackedy;
    int layerthickness;
    int itelayer;
    int preremainpy;
    int bestite;
    int packednumbox;
    int bestpackednum;
    int uld_id;
    int wt_limit;
    int rem_wt;

    double packedvolume;
    double bestvolume;
    double totalvolume;
    double totalboxvol;
    double temp;
    double percentageused;
    double percentagepackedbox;
    double elapsedtime;
    double currCost, bestCost;

    vector<Package> boxlist;

    scrappad *scrapfirst = nullptr, *smallestz = nullptr, *scrapmemb = nullptr,
             *trash = nullptr;

    Binpack(vector<Package> &packages, ULD &uld) {
        boxlist = packages;
        boxlist.insert(boxlist.begin(), Package());
        tbn = packages.size();

        xx = uld.length;
        yy = uld.width;
        zz = uld.height;
        wt_limit = uld.maxWeight;
        rem_wt = uld.maxWeight;
        uld_id = uld.id;

        totalvolume = xx * yy * zz;
        totalboxvol = 0.0;
        for (int i = 1; i <= tbn; i++) {
            totalboxvol = totalboxvol + boxlist[i].volume;
        }

        scrapfirst = new scrappad;

        scrapfirst->pre = NULL;
        scrapfirst->pos = NULL;
        bestvolume = 0.0;
        bestCost = 0.0;
        packingbest = 0;
        hundredpercent = 0;
        itenum = 0;

        execiterations();
        report();

        packages = boxlist;
        packages.erase(packages.begin());
    }

    bool allPacked() { return (bestpackednum == tbn); }

    int packlayer(void) {
        int lenx, lenz, lpz;

        if (!layerthickness) {
            packing = 0;
            return 0;
        }

        scrapfirst->cumx = px;
        scrapfirst->cumz = 0;

        while (true) {
            findsmallestz();

            if (!smallestz->pre && !smallestz->pos) {
                //*** SITUATION-1: NO BOXES ON THE RIGHT AND LEFT SIDES ***

                lenx = smallestz->cumx;
                lpz = remainpz - smallestz->cumz;
                findbox(lenx, layerthickness, remainpy, lpz, lpz);
                checkfound();

                if (layerdone)
                    break;
                if (evened)
                    continue;

                boxlist[cboxi].position.x0 = 0;
                boxlist[cboxi].position.y0 = packedy;
                boxlist[cboxi].position.z0 = smallestz->cumz;
                if (cboxx == smallestz->cumx) {
                    smallestz->cumz = smallestz->cumz + cboxz;
                } else {
                    smallestz->pos = new scrappad();
                    if (smallestz->pos == NULL) {
                        printf("Insufficient memory available\n");
                        return 1;
                    }
                    smallestz->pos->pos = NULL;
                    smallestz->pos->pre = smallestz;
                    smallestz->pos->cumx = smallestz->cumx;
                    smallestz->pos->cumz = smallestz->cumz;
                    smallestz->cumx = cboxx;
                    smallestz->cumz = smallestz->cumz + cboxz;
                }
                volumecheck();
            } else if (!smallestz->pre) {
                //*** SITUATION-2: NO BOXES ON THE LEFT SIDE ***

                lenx = smallestz->cumx;
                lenz = smallestz->pos->cumz - smallestz->cumz;
                lpz = remainpz - smallestz->cumz;
                findbox(lenx, layerthickness, remainpy, lenz, lpz);
                checkfound();

                if (layerdone)
                    break;
                if (evened)
                    continue;

                boxlist[cboxi].position.y0 = packedy;
                boxlist[cboxi].position.z0 = smallestz->cumz;
                if (cboxx == smallestz->cumx) {
                    boxlist[cboxi].position.x0 = 0;
                    if (smallestz->cumz + cboxz == smallestz->pos->cumz) {
                        smallestz->cumz = smallestz->pos->cumz;
                        smallestz->cumx = smallestz->pos->cumx;
                        trash = smallestz->pos;
                        smallestz->pos = smallestz->pos->pos;
                        if (smallestz->pos) {
                            smallestz->pos->pre = smallestz;
                        }
                        delete trash;
                    } else {
                        smallestz->cumz = smallestz->cumz + cboxz;
                    }
                } else {
                    boxlist[cboxi].position.x0 = smallestz->cumx - cboxx;
                    if (smallestz->cumz + cboxz == smallestz->pos->cumz) {
                        smallestz->cumx = smallestz->cumx - cboxx;
                    } else {
                        smallestz->pos->pre = new scrappad();
                        if (smallestz->pos->pre == NULL) {
                            printf("Insufficient memory available\n");
                            return 1;
                        }
                        smallestz->pos->pre->pos = smallestz->pos;
                        smallestz->pos->pre->pre = smallestz;
                        smallestz->pos = smallestz->pos->pre;
                        smallestz->pos->cumx = smallestz->cumx;
                        smallestz->cumx = smallestz->cumx - cboxx;
                        smallestz->pos->cumz = smallestz->cumz + cboxz;
                    }
                }
                volumecheck();
            } else if (!smallestz->pos) {
                //*** SITUATION-3: NO BOXES ON THE RIGHT SIDE ***

                lenx = smallestz->cumx - smallestz->pre->cumx;
                lenz = smallestz->pre->cumz - smallestz->cumz;
                lpz = remainpz - smallestz->cumz;
                findbox(lenx, layerthickness, remainpy, lenz, lpz);
                checkfound();

                if (layerdone)
                    break;
                if (evened)
                    continue;

                boxlist[cboxi].position.y0 = packedy;
                boxlist[cboxi].position.z0 = smallestz->cumz;
                boxlist[cboxi].position.x0 = smallestz->pre->cumx;

                if (cboxx == smallestz->cumx - smallestz->pre->cumx) {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->pre->cumx = smallestz->cumx;
                        smallestz->pre->pos = NULL;
                        delete smallestz;
                    } else {
                        smallestz->cumz = smallestz->cumz + cboxz;
                    }
                } else {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->pre->cumx = smallestz->pre->cumx + cboxx;
                    } else {
                        smallestz->pre->pos = new struct scrappad;
                        if (smallestz->pre->pos == NULL) {
                            printf("Insufficient memory available\n");
                            return 1;
                        }
                        smallestz->pre->pos->pre = smallestz->pre;
                        smallestz->pre->pos->pos = smallestz;
                        smallestz->pre = smallestz->pre->pos;
                        smallestz->pre->cumx =
                            smallestz->pre->pre->cumx + cboxx;
                        smallestz->pre->cumz = smallestz->cumz + cboxz;
                    }
                }
                volumecheck();
            } else if (smallestz->pre->cumz == smallestz->pos->cumz) {
                //*** SITUATION-4: THERE ARE BOXES ON BOTH OF THE SIDES ***

                //*** SUBSITUATION-4A: SIDES ARE EQUAL TO EACH OTHER ***

                lenx = smallestz->cumx - smallestz->pre->cumx;
                lenz = smallestz->pre->cumz - smallestz->cumz;
                lpz = remainpz - smallestz->cumz;

                findbox(lenx, layerthickness, remainpy, lenz, lpz);
                checkfound();

                if (layerdone)
                    break;
                if (evened)
                    continue;

                boxlist[cboxi].position.y0 = packedy;
                boxlist[cboxi].position.z0 = smallestz->cumz;
                if (cboxx == smallestz->cumx - smallestz->pre->cumx) {
                    boxlist[cboxi].position.x0 = smallestz->pre->cumx;
                    if (smallestz->cumz + cboxz == smallestz->pos->cumz) {
                        smallestz->pre->cumx = smallestz->pos->cumx;
                        if (smallestz->pos->pos) {
                            smallestz->pre->pos = smallestz->pos->pos;
                            smallestz->pos->pos->pre = smallestz->pre;
                            delete smallestz;
                        } else {
                            smallestz->pre->pos = NULL;
                            delete smallestz;
                        }
                    } else {
                        smallestz->cumz = smallestz->cumz + cboxz;
                    }
                } else if (smallestz->pre->cumx < px - smallestz->cumx) {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->cumx = smallestz->cumx - cboxx;
                        boxlist[cboxi].position.x0 = smallestz->cumx - cboxx;
                    } else {
                        boxlist[cboxi].position.x0 = smallestz->pre->cumx;
                        smallestz->pre->pos = new scrappad();
                        if (smallestz->pre->pos == NULL) {
                            printf("Insufficient memory available\n");
                            return 1;
                        }
                        smallestz->pre->pos->pre = smallestz->pre;
                        smallestz->pre->pos->pos = smallestz;
                        smallestz->pre = smallestz->pre->pos;
                        smallestz->pre->cumx =
                            smallestz->pre->pre->cumx + cboxx;
                        smallestz->pre->cumz = smallestz->cumz + cboxz;
                    }
                } else {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->pre->cumx = smallestz->pre->cumx + cboxx;
                        boxlist[cboxi].position.x0 = smallestz->pre->cumx;
                    } else {
                        boxlist[cboxi].position.x0 = smallestz->cumx - cboxx;
                        smallestz->pos->pre = new scrappad();
                        if (smallestz->pos->pre == NULL) {
                            printf("Insufficient memory available\n");
                            return 1;
                        }
                        smallestz->pos->pre->pos = smallestz->pos;
                        smallestz->pos->pre->pre = smallestz;
                        smallestz->pos = smallestz->pos->pre;
                        smallestz->pos->cumx = smallestz->cumx;
                        smallestz->pos->cumz = smallestz->cumz + cboxz;
                        smallestz->cumx = smallestz->cumx - cboxx;
                    }
                }
                volumecheck();
            } else {
                //*** SUBSITUATION-4B: SIDES ARE NOT EQUAL TO EACH OTHER ***

                lenx = smallestz->cumx - smallestz->pre->cumx;
                lenz = smallestz->pre->cumz - smallestz->cumz;
                lpz = remainpz - smallestz->cumz;
                findbox(lenx, layerthickness, remainpy, lenz, lpz);
                checkfound();

                if (layerdone)
                    break;
                if (evened)
                    continue;

                boxlist[cboxi].position.y0 = packedy;
                boxlist[cboxi].position.z0 = smallestz->cumz;
                boxlist[cboxi].position.x0 = smallestz->pre->cumx;
                if (cboxx == smallestz->cumx - smallestz->pre->cumx) {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->pre->cumx = smallestz->cumx;
                        smallestz->pre->pos = smallestz->pos;
                        smallestz->pos->pre = smallestz->pre;
                        delete smallestz;
                    } else {
                        smallestz->cumz = smallestz->cumz + cboxz;
                    }
                } else {
                    if (smallestz->cumz + cboxz == smallestz->pre->cumz) {
                        smallestz->pre->cumx = smallestz->pre->cumx + cboxx;
                    } else if (smallestz->cumz + cboxz ==
                               smallestz->pos->cumz) {
                        boxlist[cboxi].position.x0 = smallestz->cumx - cboxx;
                        smallestz->cumx = smallestz->cumx - cboxx;
                    } else {
                        smallestz->pre->pos = new scrappad();
                        if (smallestz->pre->pos == NULL) {
                            printf("Insufficient memory available\n");
                            return 1;
                        }
                        smallestz->pre->pos->pre = smallestz->pre;
                        smallestz->pre->pos->pos = smallestz;
                        smallestz->pre = smallestz->pre->pos;
                        smallestz->pre->cumx =
                            smallestz->pre->pre->cumx + cboxx;
                        smallestz->pre->cumz = smallestz->cumz + cboxz;
                    }
                }
                volumecheck();
            }
        }
        return 0;
    }

    void execiterations() {
        for (int variant = 1; (variant <= 6);
             variant++) // single orientation of bin
        {
            switch (variant) {
            case 1:
                px = xx;
                py = yy;
                pz = zz;
                break;
            case 2:
                px = zz;
                py = yy;
                pz = xx;
                break;
            case 3:
                px = zz;
                py = xx;
                pz = yy;
                break;
            case 4:
                px = yy;
                py = xx;
                pz = zz;
                break;
            case 5:
                px = xx;
                py = zz;
                pz = yy;
                break;
            case 6:
                px = yy;
                py = zz;
                pz = xx;
                break;
            }

            listcanditlayers();
            layers[0].layereval = -1;
            qsort(layers, layerlistlen, sizeof(layers[0]), complayerlist);

            for (layersindex = 1; (layersindex <= layerlistlen);
                 layersindex++) {
                ++itenum;

                packedvolume = 0.0;
                currCost = 0.0;
                packedy = 0;
                packing = 1;
                layerthickness = layers[layersindex].layerdim;
                itelayer = layersindex;
                remainpy = py;
                remainpz = pz;
                packednumbox = 0;

                // Reset boxes' packing status
                for (int x = 1; x <= tbn; x++) {
                    boxlist[x].isPacked = 0; // Reset packing status to 0
                }
                rem_wt = wt_limit;

                // BEGIN DO-WHILE
                do {
                    layerinlayer = 0;
                    layerdone = 0;

                    if (packlayer()) {
                        return; // Replace exit(1) with return to properly exit
                                // the function
                    }

                    packedy += layerthickness;
                    remainpy = py - packedy;

                    if (layerinlayer) {
                        prepackedy = packedy;
                        preremainpy = remainpy;
                        remainpy = layerthickness - prelayer;
                        packedy = packedy - layerthickness + prelayer;
                        remainpz = lilz;
                        layerthickness = layerinlayer;
                        layerdone = 0;

                        if (packlayer()) {
                            return; // Replace exit(1) with return to properly
                                    // exit the function
                        }

                        packedy = prepackedy;
                        remainpy = preremainpy;
                        remainpz = pz;
                    }

                    findlayer(remainpy);
                } while (packing);
                // END DO-WHILE

                if ((currCost > bestCost) ||
                    (currCost == bestCost && packedvolume > bestvolume)) {
                    bestvolume = packedvolume;
                    bestvariant = variant;
                    bestite = itelayer;
                    bestpackednum = packednumbox;
                    bestCost = currCost;
                }

                if (hundredpercent)
                    break;

                percentageused = bestvolume * 100 / totalvolume;
            }

            if (hundredpercent)
                break;

            if ((xx == yy) && (yy == zz))
                variant = 6;
        }
    }
    void volumecheck(void) {
        boxlist[cboxi].isPacked = 1;
        boxlist[cboxi].packx = cboxx;
        boxlist[cboxi].packy = cboxy;
        boxlist[cboxi].packz = cboxz;
        boxlist[cboxi].uldID = uld_id;
        rem_wt -= boxlist[cboxi].weight;
        packedvolume += boxlist[cboxi].volume;

        if (boxlist[cboxi].type == 'E')
            currCost += boxlist[cboxi].extraCost;
        else
            currCost += inf;

        packednumbox++;
        if (packingbest) {
            outputboxlist();
        } else if (packedvolume == totalvolume || packedvolume == totalboxvol) {
            packing = 0;
            hundredpercent = 1;
        }
        return;
    }

    int findlayer(int thickness) {
        int exdim, dimdif, dimen2, dimen3, y, z;
        int layereval, eval;
        layerthickness = 0;
        eval = 1000000;
        for (x = 1; x <= tbn; x++) {
            if (boxlist[x].isPacked)
                continue;
            for (y = 1; y <= 3; y++) {
                switch (y) {
                case 1:
                    exdim = boxlist[x].length;
                    dimen2 = boxlist[x].width;
                    dimen3 = boxlist[x].height;
                    break;
                case 2:
                    exdim = boxlist[x].width;
                    dimen2 = boxlist[x].length;
                    dimen3 = boxlist[x].height;
                    break;
                case 3:
                    exdim = boxlist[x].height;
                    dimen2 = boxlist[x].length;
                    dimen3 = boxlist[x].width;
                    break;
                }
                layereval = 0;
                if ((exdim <= thickness) &&
                    (((dimen2 <= px) && (dimen3 <= pz)) ||
                     ((dimen3 <= px) && (dimen2 <= pz)))) {
                    for (z = 1; z <= tbn; z++) {
                        if (!(x == z) && !(boxlist[z].isPacked)) {
                            dimdif = abs(exdim - boxlist[z].length);
                            if (abs(exdim - boxlist[z].width) < dimdif) {
                                dimdif = abs(exdim - boxlist[z].width);
                            }
                            if (abs(exdim - boxlist[z].height) < dimdif) {
                                dimdif = abs(exdim - boxlist[z].height);
                            }
                            layereval = layereval + dimdif;
                        }
                    }
                    if (layereval < eval) {
                        eval = layereval;
                        layerthickness = exdim;
                    }
                }
            }
        }
        if (layerthickness == 0 || layerthickness > remainpy)
            packing = 0;
        return 0;
    }

    void listcanditlayers() {
        int same;
        int exdim, dimdif, dimen2, dimen3, y, z, k;
        int layereval;

        layerlistlen = 0;

        for (x = 1; x <= tbn; x++) {
            for (y = 1; y <= 3; y++) {
                switch (y) {
                case 1:
                    exdim = boxlist[x].length;
                    dimen2 = boxlist[x].width;
                    dimen3 = boxlist[x].height;
                    break;
                case 2:
                    exdim = boxlist[x].width;
                    dimen2 = boxlist[x].length;
                    dimen3 = boxlist[x].height;
                    break;
                case 3:
                    exdim = boxlist[x].height;
                    dimen2 = boxlist[x].length;
                    dimen3 = boxlist[x].width;
                    break;
                }
                if ((exdim > py) || (((dimen2 > px) || (dimen3 > pz)) &&
                                     ((dimen3 > px) || (dimen2 > pz))))
                    continue;
                same = 0;

                for (k = 1; k <= layerlistlen; k++) {
                    if (exdim == layers[k].layerdim) {
                        same = 1;
                        continue;
                    }
                }
                if (same)
                    continue;
                layereval = 0;
                for (z = 1; z <= tbn; z++) {
                    if (!(x == z)) {
                        dimdif = abs(exdim - boxlist[z].length);
                        if (abs(exdim - boxlist[z].width) < dimdif) {
                            dimdif = abs(exdim - boxlist[z].width);
                        }
                        if (abs(exdim - boxlist[z].height) < dimdif) {
                            dimdif = abs(exdim - boxlist[z].height);
                        }
                        layereval = layereval + dimdif;
                    }
                }
                layers[++layerlistlen].layereval = layereval;
                layers[layerlistlen].layerdim = exdim;
            }
        }
        return;
    }

    void outputboxlist(void) {
        string strx;
        string strisPacked;
        string strdim1, strdim2, strdim3;
        string strcox, strcoy, strcoz;
        string strpackx, strpacky, strpackz;

        int x, y, z, bx, by, bz;

        switch (bestvariant) {
        case 1:
            x = boxlist[cboxi].position.x0;
            y = boxlist[cboxi].position.y0;
            z = boxlist[cboxi].position.z0;
            bx = boxlist[cboxi].packx;
            by = boxlist[cboxi].packy;
            bz = boxlist[cboxi].packz;
            break;
        case 2:
            x = boxlist[cboxi].position.z0;
            y = boxlist[cboxi].position.y0;
            z = boxlist[cboxi].position.x0;
            bx = boxlist[cboxi].packz;
            by = boxlist[cboxi].packy;
            bz = boxlist[cboxi].packx;
            break;
        case 3:
            x = boxlist[cboxi].position.y0;
            y = boxlist[cboxi].position.z0;
            z = boxlist[cboxi].position.x0;
            bx = boxlist[cboxi].packy;
            by = boxlist[cboxi].packz;
            bz = boxlist[cboxi].packx;
            break;
        case 4:
            x = boxlist[cboxi].position.y0;
            y = boxlist[cboxi].position.x0;
            z = boxlist[cboxi].position.z0;
            bx = boxlist[cboxi].packy;
            by = boxlist[cboxi].packx;
            bz = boxlist[cboxi].packz;
            break;
        case 5:
            x = boxlist[cboxi].position.x0;
            y = boxlist[cboxi].position.z0;
            z = boxlist[cboxi].position.y0;
            bx = boxlist[cboxi].packx;
            by = boxlist[cboxi].packz;
            bz = boxlist[cboxi].packy;
            break;
        case 6:
            x = boxlist[cboxi].position.z0;
            y = boxlist[cboxi].position.x0;
            z = boxlist[cboxi].position.y0;
            bx = boxlist[cboxi].packz;
            by = boxlist[cboxi].packx;
            bz = boxlist[cboxi].packy;
            break;
        }

        boxlist[cboxi].position.x0 = x;
        boxlist[cboxi].position.y0 = y;
        boxlist[cboxi].position.z0 = z;
        boxlist[cboxi].packx = bx;
        boxlist[cboxi].packy = by;
        boxlist[cboxi].packz = bz;
        boxlist[cboxi].position.x1 = x + bx;
        boxlist[cboxi].position.y1 = y + by;
        boxlist[cboxi].position.z1 = z + bz;

        return;
    }

    void findsmallestz() {
        scrapmemb = scrapfirst;
        smallestz = scrapmemb;

        // Traverse the linked list to find the node with the smallest `cumz`
        while (scrapmemb->pos != nullptr) {
            if (scrapmemb->pos->cumz < smallestz->cumz) {
                smallestz = scrapmemb->pos;
            }
            scrapmemb = scrapmemb->pos;
        }
    }

    void findbox(int hmx, int hy, int hmy, int hz, int hmz) {
        bfx = inf;
        bfy = inf;
        bfz = inf;
        bbfx = inf;
        bbfy = inf;
        bbfz = inf;
        boxi = 0;
        bboxi = 0;
        for (x = 1; x <= tbn; x++) {

            if (boxlist[x].isPacked)
                continue;
            if (boxlist[x].weight > rem_wt)
                continue;

            if (x > tbn)
                return;
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].length,
                       boxlist[x].width, boxlist[x].height);
            if ((boxlist[x].length == boxlist[x].height) &&
                (boxlist[x].height == boxlist[x].width))
                continue;
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].length,
                       boxlist[x].height, boxlist[x].width);
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].width,
                       boxlist[x].length, boxlist[x].height);
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].width,
                       boxlist[x].height, boxlist[x].length);
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].height,
                       boxlist[x].length, boxlist[x].width);
            analyzebox(hmx, hy, hmy, hz, hmz, boxlist[x].height,
                       boxlist[x].width, boxlist[x].length);
        }
    }

    void analyzebox(int hmx, int hy, int hmy, int hz, int hmz, int dim1,
                    int dim2, int dim3) {
        if (dim1 <= hmx && dim2 <= hmy && dim3 <= hmz) {
            if (dim2 <= hy) {
                if (hy - dim2 < bfy) {
                    boxx = dim1;
                    boxy = dim2;
                    boxz = dim3;
                    bfx = hmx - dim1;
                    bfy = hy - dim2;
                    bfz = abs(hz - dim3);
                    boxi = x;
                } else if (hy - dim2 == bfy && hmx - dim1 < bfx) {
                    boxx = dim1;
                    boxy = dim2;
                    boxz = dim3;
                    bfx = hmx - dim1;
                    bfy = hy - dim2;
                    bfz = abs(hz - dim3);
                    boxi = x;
                } else if (hy - dim2 == bfy && hmx - dim1 == bfx &&
                           abs(hz - dim3) < bfz) {
                    boxx = dim1;
                    boxy = dim2;
                    boxz = dim3;
                    bfx = hmx - dim1;
                    bfy = hy - dim2;
                    bfz = abs(hz - dim3);
                    boxi = x;
                }
            } else {
                if (dim2 - hy < bbfy) {
                    bboxx = dim1;
                    bboxy = dim2;
                    bboxz = dim3;
                    bbfx = hmx - dim1;
                    bbfy = dim2 - hy;
                    bbfz = abs(hz - dim3);
                    bboxi = x;
                } else if (dim2 - hy == bbfy && hmx - dim1 < bbfx) {
                    bboxx = dim1;
                    bboxy = dim2;
                    bboxz = dim3;
                    bbfx = hmx - dim1;
                    bbfy = dim2 - hy;
                    bbfz = abs(hz - dim3);
                    bboxi = x;
                } else if (dim2 - hy == bbfy && hmx - dim1 == bbfx &&
                           abs(hz - dim3) < bbfz) {
                    bboxx = dim1;
                    bboxy = dim2;
                    bboxz = dim3;
                    bbfx = hmx - dim1;
                    bbfy = dim2 - hy;
                    bbfz = abs(hz - dim3);
                    bboxi = x;
                }
            }
        }
    }

    void checkfound() {
        evened = 0;
        if (boxi) {
            cboxi = boxi;
            cboxx = boxx;
            cboxy = boxy;
            cboxz = boxz;
        } else {
            if ((bboxi > 0) &&
                (layerinlayer || (!smallestz->pre && !smallestz->pos))) {
                if (!layerinlayer) {
                    prelayer = layerthickness;
                    lilz = smallestz->cumz;
                }
                cboxi = bboxi;
                cboxx = bboxx;
                cboxy = bboxy;
                cboxz = bboxz;
                layerinlayer = layerinlayer + bboxy - layerthickness;
                layerthickness = bboxy;
            } else {
                if (!smallestz->pre && !smallestz->pos) {
                    layerdone = 1;
                } else {
                    evened = 1;
                    if (!smallestz->pre) {
                        trash = smallestz->pos;
                        smallestz->cumx = smallestz->pos->cumx;
                        smallestz->cumz = smallestz->pos->cumz;
                        smallestz->pos = smallestz->pos->pos;
                        if (smallestz->pos) {
                            smallestz->pos->pre = smallestz;
                        }
                        delete trash;
                    } else if (!smallestz->pos) {
                        smallestz->pre->pos = NULL;
                        smallestz->pre->cumx = smallestz->cumx;
                        delete smallestz;
                    } else {
                        if (smallestz->pre->cumz == smallestz->pos->cumz) {
                            smallestz->pre->pos = smallestz->pos->pos;
                            if (smallestz->pos->pos) {
                                smallestz->pos->pos->pre = smallestz->pre;
                            }
                            smallestz->pre->cumx = smallestz->pos->cumx;
                            delete smallestz->pos;
                            delete smallestz;
                        } else {
                            smallestz->pre->pos = smallestz->pos;
                            smallestz->pos->pre = smallestz->pre;
                            if (smallestz->pre->cumz < smallestz->pos->cumz) {
                                smallestz->pre->cumx = smallestz->cumx;
                            }
                            delete smallestz;
                        }
                    }
                }
            }
        }
        return;
    }

    void report(void) {
        switch (bestvariant) {
        case 1:
            px = xx;
            py = yy;
            pz = zz;
            break;
        case 2:
            px = zz;
            py = yy;
            pz = xx;
            break;
        case 3:
            px = zz;
            py = xx;
            pz = yy;
            break;
        case 4:
            px = yy;
            py = xx;
            pz = zz;
            break;
        case 5:
            px = xx;
            py = zz;
            pz = yy;
            break;
        case 6:
            px = yy;
            py = zz;
            pz = xx;
            break;
        }
        packingbest = 1;

        strpx = to_string(px);
        strpy = to_string(py);
        strpz = to_string(pz);

        percentagepackedbox = bestvolume * 100 / totalboxvol;
        percentageused = bestvolume * 100 / totalvolume;

        listcanditlayers();
        layers[0].layereval = -1;
        qsort(layers, layerlistlen, sizeof(layers[0]), complayerlist);
        packedvolume = 0.0;
        currCost = 0.0;
        packedy = 0;
        packing = 1;
        layerthickness = layers[bestite].layerdim;
        remainpy = py;
        remainpz = pz;

        for (x = 1; x <= tbn; x++) {
            boxlist[x].isPacked = 0;
        }
        rem_wt = wt_limit;

        do {
            layerinlayer = 0;
            layerdone = 0;
            packlayer();
            packedy = packedy + layerthickness;
            remainpy = py - packedy;
            if (layerinlayer) {
                prepackedy = packedy;
                preremainpy = remainpy;
                remainpy = layerthickness - prelayer;
                packedy = packedy - layerthickness + prelayer;
                remainpz = lilz;
                layerthickness = layerinlayer;
                layerdone = 0;
                packlayer();
                packedy = prepackedy;
                remainpy = preremainpy;
                remainpz = pz;
            }

            findlayer(remainpy);

        } while (packing);
    }
};