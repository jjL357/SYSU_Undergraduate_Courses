package com.example.demo.controller;

import com.example.demo.model.Post;
import com.example.demo.model.User;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import com.example.demo.service.PostService;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
@Controller
public class LoginController {

    @Autowired
    private UserService userService;

    @Autowired
    private PostService postService;

    @GetMapping("/login")
    public String login() {
        return "login";
    }

    private static final String UPLOAD_DIR = "src/main/resources/static/posts_photos";

    @PostMapping("/login")
    public String loginSubmit(@RequestParam String uname,
                              @RequestParam String psw,
                              HttpSession session,
                              Model model) {
        User user = userService.findUserByNameAndPassword(uname, psw);
        if (user != null) {
            session.setAttribute("user", user);
            if(user.getIsAdmin()==1)return "redirect:/admin";
            else return "redirect:/userProfile";
        } else {
            model.addAttribute("error", "用户名或密码不正确，请重试。");
            return "login";
        }
    }

    @GetMapping("/admin")
    public String AdminLogin(HttpSession session,
                              Model model) {
        User user = (User) session.getAttribute("user");
        if (user == null) {
            return "redirect:/login";
        }
        Long authorId = user.getUid();

        List<Post> posts = postService.getAllPosts();
        model.addAttribute("posts", posts);
        model.addAttribute("user", user);

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
        return "admin";
    }

    @PostMapping("/admin/deletePost/{postId}")
    public String deletePost(@PathVariable Integer postId, HttpSession session, Model model) {
        User user = (User) session.getAttribute("user");
        session.setAttribute("user", user);
        // Check if the post belongs to the logged-in user
        Post post = postService.getPostByPostId(postId);
        if (post == null ) {
            model.addAttribute("error", "无法删除帖子！");
            return "redirect:/admin";
        }

        // Delete post and related photos
        postService.deletePost(postId);
        deletePostPhotos(postId);

        model.addAttribute("message", "帖子删除成功！");
        return "redirect:/admin";
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

    

    @GetMapping("/userProfile")
    public String userProfile(Model model, HttpSession session) {
        User user = (User) session.getAttribute("user");
        List<Post> posts = postService.getAllPosts();
        List<Post> hotPosts = postService.findHotPosts(); // 假设这里是获取点赞率最高的15个帖子的方法
        model.addAttribute("hotPosts", hotPosts);
        model.addAttribute("posts", posts);
        if (user != null) {
            model.addAttribute("user", user);
            return "userProfile";
        } else {
            return "redirect:/login";
        }
    }

    @PostMapping("/uploadAvatar")
    public String uploadAvatar(@RequestPart("avatar") MultipartFile avatar,
                               HttpSession session) throws IOException {
        User user = (User) session.getAttribute("user");
        if (user != null && !avatar.isEmpty()) {
            String filename = user.getUid() + ".jpg";
            String uploadDir = ("D:\\d_code\\git\\软件工程\\test\\demo\\src\\main\\resources\\static\\avatar").toString();
            File file = new File(uploadDir, filename);
            avatar.transferTo(file);
            user.setAvatar(filename);
            userService.updateUser(user); // Ensure this method exists to update user info
        }
        return "redirect:/userProfile";
    }
}
