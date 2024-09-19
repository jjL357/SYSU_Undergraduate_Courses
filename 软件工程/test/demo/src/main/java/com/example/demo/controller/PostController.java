package com.example.demo.controller;

import com.example.demo.model.Post;
import com.example.demo.model.User;
import com.example.demo.service.PostService;
import com.example.demo.service.UserService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;

import javax.servlet.http.HttpSession;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;

@Controller
public class PostController {

    @Autowired
    private PostService postService;

    @Autowired
    private UserService userService;

    private static final String UPLOAD_DIR = "src/main/resources/static/posts_photos";

    @GetMapping("/createPost")
    public String showCreatePostPage(HttpSession session) {
        User user = (User) session.getAttribute("user");
        if (user == null) {
            return "redirect:/login";
        }
        return "createPost";
    }

    @PostMapping("/createPost")
    public String createPost(@RequestParam String title,
                             @RequestParam String content,
                             @RequestParam("files") List<MultipartFile> files,
                             HttpSession session,
                             Model model) throws IOException {
        // 获取当前用户的ID
        User user = (User) session.getAttribute("user");
        if (user == null) {
            return "redirect:/login";
        }
        Long authorId = user.getUid();

        Post post = new Post();
        post.setTitle(title);
        post.setContent(content);
        post.setAuthorId(authorId);

        Post savedPost = postService.savePost(post);

        // 保存上传的照片
        saveFiles(savedPost.getPostId(), files);

        model.addAttribute("message", "帖子发布成功！");
        return "redirect:/myPosts";
    }

    private void saveFiles(Integer postId, List<MultipartFile> files) throws IOException {
        String postDir = UPLOAD_DIR + File.separator + postId;
        Path postDirPath = Paths.get(postDir);
        if (!Files.exists(postDirPath)) {
            Files.createDirectories(postDirPath);
        }

        int count = 1;
        for (MultipartFile file : files) {
            if (count > 9) {
                break;  // 最多上传9张照片
            }

            if (!file.isEmpty()) {
                String originalFilename = file.getOriginalFilename();
                String extension = originalFilename.substring(originalFilename.lastIndexOf("."));
                String filename = UUID.randomUUID().toString() + extension;

                Path filePath = Paths.get(postDir, filename);
                Files.write(filePath, file.getBytes());

                // Increment count after successful upload
                count++;
            }
        }
    }

    @GetMapping("/myPosts")
    public String showMyPosts(Model model, HttpSession session) {
        User user = (User) session.getAttribute("user");
        if (user == null) {
            return "redirect:/login";
        }
        Long authorId = user.getUid();

        List<Post> posts = postService.getPostsByAuthorId(authorId);
        model.addAttribute("posts", posts);

        // 构造帖子的图片路径 Map
        Map<Integer, List<String>> postPhotos = new HashMap<>();
        for (Post post : posts) {
            List<String> photos = new ArrayList<>();
            File postFolder = new File(UPLOAD_DIR + File.separator + post.getPostId());
            if (postFolder.exists() && postFolder.isDirectory()) {
                File[] files = postFolder.listFiles();
                if (files != null) {
                    for (File file : files) {
                        photos.add(file.getName());
                    }
                }
            }
            postPhotos.put(post.getPostId(), photos);
        }
        model.addAttribute("postPhotos", postPhotos);

        return "myPosts";
    }

    @PostMapping("/deletePost/{postId}")
    public String deletePost(@PathVariable Integer postId, HttpSession session, Model model) {
        User user = (User) session.getAttribute("user");
        
        // Check if the post belongs to the logged-in user
        Post post = postService.getPostByPostId(postId);
        if (post == null ) {
            model.addAttribute("error", "无法删除帖子！");
            return "redirect:/myPosts";
        }

        // Delete post and related photos
        postService.deletePost(postId);
        deletePostPhotos(postId);

        model.addAttribute("message", "帖子删除成功！");
        return "redirect:/myPosts";
    }

    private void deletePostPhotos(Integer postId) {
        File postFolder = new File(UPLOAD_DIR + File.separator + postId);
        if (postFolder.exists() && postFolder.isDirectory()) {
            File[] files = postFolder.listFiles();
            if (files != null) {
                for (File file : files) {
                    file.delete();
                }
            }
            postFolder.delete();
        }
    }

    @GetMapping("/post/{postId}")
    public String viewPost(@PathVariable Integer postId, Model model) {
        Post post = postService.getPostByPostId(postId);
        if (post == null) {
            return "redirect:/error"; // 处理帖子不存在的情况
        }
    
        Long authorId = post.getAuthorId();
        User user = userService.findUserByUid(authorId); // 确保 UserService 有这个方法
        if (user == null) {
            return "redirect:/error"; // 处理用户不存在的情况
        }
    
        model.addAttribute("user", user);
        model.addAttribute("post", post);
    
        // 构造帖子的图片路径 Map
        Map<Integer, List<String>> postPhotos = new HashMap<>();
        List<String> photos = new ArrayList<>();
        File postFolder = new File(UPLOAD_DIR + File.separator + post.getPostId());
        if (postFolder.exists() && postFolder.isDirectory()) {
            File[] files = postFolder.listFiles();
            if (files != null) {
                for (File file : files) {
                    photos.add(file.getName());
                }
            }
        }
        postPhotos.put(post.getPostId(), photos);
        model.addAttribute("postPhotos", postPhotos);
    
        return "postDetail";
    }
    

}
