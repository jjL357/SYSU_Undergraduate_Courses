package com.example.demo.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/music")
public class PlayController {

    private boolean isPlaying = false;

    @PostMapping("/play")
    public void playMusic() {
        isPlaying = true;
        // 实际上可以在这里调用音乐播放的相关逻辑
        System.out.println("音乐已开始播放");
    }

    @PostMapping("/pause")
    public void pauseMusic() {
        isPlaying = false;
        // 实际上可以在这里调用音乐暂停的相关逻辑
        System.out.println("音乐已暂停");
    }

    @GetMapping("/status")
    public boolean getMusicStatus() {
        return isPlaying;
    }
}
