(() => {
    const videos = Array.from(document.querySelectorAll('video[data-sync-group="cars"]'));

    if (videos.length < 2) {
        return;
    }

    const master = videos.find((video) => video.hasAttribute("data-sync-master")) || videos[0];
    const followers = videos.filter((video) => video !== master);
    const softTolerance = 0.035;
    const hardTolerance = 0.12;
    const maximumRateAdjustment = 0.03;
    let temporarilyPausingFollowers = false;

    const waitUntilPlayable = (video) => {
        if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
            return Promise.resolve();
        }

        return new Promise((resolve) => {
            const finish = () => {
                video.removeEventListener("canplay", finish);
                video.removeEventListener("error", finish);
                resolve();
            };

            video.addEventListener("canplay", finish, { once: true });
            video.addEventListener("error", finish, { once: true });
        });
    };

    const circularDifference = (time, reference, duration) => {
        let difference = time - reference;

        if (Number.isFinite(duration) && duration > 0) {
            if (difference > duration / 2) {
                difference -= duration;
            } else if (difference < -duration / 2) {
                difference += duration;
            }
        }

        return difference;
    };

    const alignFollower = (follower) => {
        if (follower.readyState < HTMLMediaElement.HAVE_METADATA) {
            return;
        }

        follower.currentTime = master.currentTime;
        follower.playbackRate = master.playbackRate;
    };

    const playFollower = (follower) => {
        const playAttempt = follower.play();

        if (playAttempt) {
            playAttempt.catch(() => {
                // Muted autoplay may still be blocked in power-saving or background modes.
                // The next master play event or visibility change retries playback.
            });
        }
    };

    const hardAlignAll = ({ play = !master.paused } = {}) => {
        followers.forEach((follower) => {
            alignFollower(follower);

            if (play) {
                playFollower(follower);
            }
        });
    };

    const pauseFollowers = () => {
        temporarilyPausingFollowers = true;
        followers.forEach((follower) => {
            follower.pause();
            follower.playbackRate = master.playbackRate;
        });
        temporarilyPausingFollowers = false;
    };

    const correctDrift = () => {
        if (master.paused || master.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            pauseFollowers();
            document.documentElement.dataset.carVideoMaxDriftMs = "0";
            return;
        }

        const duration = master.duration;
        let maximumDrift = 0;

        followers.forEach((follower) => {
            if (follower.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                return;
            }

            if (follower.paused && !temporarilyPausingFollowers) {
                alignFollower(follower);
                playFollower(follower);
                return;
            }

            const drift = circularDifference(follower.currentTime, master.currentTime, duration);
            maximumDrift = Math.max(maximumDrift, Math.abs(drift));

            if (Math.abs(drift) >= hardTolerance) {
                alignFollower(follower);
                return;
            }

            if (Math.abs(drift) >= softTolerance) {
                const correction = Math.max(
                    -maximumRateAdjustment,
                    Math.min(maximumRateAdjustment, -drift * 0.25),
                );
                follower.playbackRate = master.playbackRate + correction;
            } else if (follower.playbackRate !== master.playbackRate) {
                follower.playbackRate = master.playbackRate;
            }
        });

        document.documentElement.dataset.carVideoMaxDriftMs = String(Math.round(maximumDrift * 1000));
    };

    const startTogether = async () => {
        await Promise.all(videos.map(waitUntilPlayable));

        videos.forEach((video) => {
            video.pause();
            video.currentTime = 0;
            video.playbackRate = master.playbackRate;
        });

        await new Promise(requestAnimationFrame);
        await Promise.allSettled(videos.map((video) => video.play()));
        hardAlignAll();
        document.documentElement.dataset.carVideoSync = "ready";
    };

    master.addEventListener("play", () => hardAlignAll({ play: true }));
    master.addEventListener("playing", () => hardAlignAll({ play: true }));
    master.addEventListener("pause", pauseFollowers);
    master.addEventListener("seeking", () => hardAlignAll({ play: false }));
    master.addEventListener("seeked", () => hardAlignAll());
    master.addEventListener("ratechange", () => {
        followers.forEach((follower) => {
            follower.playbackRate = master.playbackRate;
        });
    });
    master.addEventListener("waiting", pauseFollowers);
    master.addEventListener("stalled", pauseFollowers);

    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) {
            hardAlignAll();
        }
    });
    window.addEventListener("pageshow", () => hardAlignAll());

    window.setInterval(correctDrift, 250);
    startTogether();
})();
