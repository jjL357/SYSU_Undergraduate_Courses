package com.example.demo.controller;
import com.example.demo.repository.*;
import com.example.demo.dao.LikeDAO;
import com.example.demo.model.Post;
import com.example.demo.model.User;
import com.example.demo.model.Comment;
import com.example.demo.repository.PostRepository;
import com.example.demo.service.PostService;
import com.example.demo.service.UserService;
import com.example.demo.service.LikeService;
import com.example.demo.service.CommentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import javax.servlet.http.HttpSession;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;

@Controller
public class PostController {

    @Autowired
    private PostService postService;

    @Autowired
    private UserService userService;

    @Autowired
    private LikeService likeService;

    @Autowired
    private CommentService commentService;

    @Autowired
    private LikeDAO likeDAO;

    @Autowired
    private PostRepository postRepository;

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

        return "myPosts";
    }

    @PostMapping("/deletePost/{postId}")
    public String deletePost(@PathVariable Integer postId, HttpSession session, Model model) {
        User user = (User) session.getAttribute("user");

        // Check if the post belongs to the logged-in user
        Post post = postService.getPostByPostId(postId);
        if (post == null || !post.getAuthorId().equals(user.getUid())) {
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

    @GetMapping("/post/{postId}/{userid}")
    public String viewPost(@PathVariable Integer postId,@PathVariable Integer userid,HttpSession session, Model model) {
        User user = userService.findUserByUid(userid.longValue());
        Post post = postService.getPostByPostId(postId);
        
        if (user == null) {
            return "redirect:/error"; // 处理用户不存在的情况
        }
        Long authorId = post.getAuthorId();
        User author = userService.findUserByUid(authorId); 
        if (author == null) {
            return "redirect:/error"; // 处理用户不存在的情况
        }
        
        Long likeCount = likeService.countLikes(postId.longValue()); // 获取点赞数

        boolean likedByUser = likeService.isPostLikedByUser( user.getUid(),postId.longValue());
        model.addAttribute("author", author);
        model.addAttribute("user", user);
        model.addAttribute("post", post);
        model.addAttribute("likeCount", likeCount); // 将点赞数传递到模板
        model.addAttribute("likedByUser", likedByUser); 
        
        List <Comment> comments = commentService.getAllCommentsByPostId(postId.longValue()); 
        model.addAttribute("comments", comments); 
        
        
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
    

     @PostMapping("/likePost")
    public ResponseEntity<String> likePost(@RequestParam Long postId, @RequestParam Long userId) {
        // 根据 postId 和 userId 进行点赞或取消点赞操作
        boolean liked = likeService.toggleLike(postId, userId);

        if (liked) {
            return ResponseEntity.ok("点赞成功");
        } else {
            return ResponseEntity.ok("取消点赞成功");
        }
    }

    @GetMapping("/getLikeCount")
    public ResponseEntity<Long> getLikeCount(@RequestParam Long postId) {
        Long likeCount = likeService.countLikes(postId);
        return ResponseEntity.ok(likeCount);
    }
    

    @GetMapping("/countLikes")
    @ResponseBody
    public ResponseEntity<Long> countLikes(@RequestParam Long postId) {
        Long likeCount = likeService.countLikes(postId);
        return ResponseEntity.ok(likeCount);
    }

    @PostMapping("/processLikes/{postId}/{uid}")
    public String processLikes(@PathVariable Long postId, @PathVariable Long uid, Model model, HttpSession session) {
        // 根据 uid 查找用户
        User user = userService.findUserByUid(uid);
        if (user == null) {
            // 处理未找到用户的情况，这里可以根据实际情况处理，比如跳转到错误页面或者其他逻辑
            return "redirect:/register";
        }

        // 检查当前用户是否已经点赞过
        boolean likedByUser = likeService.isPostLikedByUser(uid,postId);
        if (likedByUser) {
            // 如果已经点赞过，则取消点赞
            likeService.unlikePost(uid,postId);
        } else {
            // 如果未点赞，则点赞
            likeService.likePost(uid,postId);
        }

        // 重定向到帖子详情页面，这里根据实际情况修改重定向的路径
       return "redirect:/post/" + postId + "/" + uid;
    }

    @GetMapping("/likedPosts")
    public String likedPosts(Model model, HttpSession session) {
        // 获取当前登录的用户
        User user = (User) session.getAttribute("user");

        if (user == null) {
            // 如果用户未登录，重定向到登录页面或其他处理
            return "redirect:/login";
        }

        // 获取用户点赞过的帖子ID列表
        List<Long> likedPostIds = likeDAO.getLikedPosts(user);

        // 获取点赞过的帖子详细信息
        List<Post> likedPosts = new ArrayList<>();
        for (Long id : likedPostIds) {
            likedPosts.add(postRepository.findByPostId(id.intValue()));
        }

        // 将点赞过的帖子列表传递到模板中
        model.addAttribute("likedPosts", likedPosts);
        model.addAttribute("user", user);
        //return "redirect:/login";
        return "like_posts"; // 返回HTML模板名称
    }

    @PostMapping("/comment/{postId}/{uid}")
    public String processComment(@RequestParam String content,@PathVariable Long postId, @PathVariable Long uid, Model model, HttpSession session) {
        // 根据 uid 查找用户
        User user = userService.findUserByUid(uid);
        if (user == null) {
            // 处理未找到用户的情况，这里可以根据实际情况处理，比如跳转到错误页面或者其他逻辑
            return "redirect:/register";
        }

        Comment comment = new Comment(postId, 0L, uid, content);
        commentService.saveComment(comment);
        
        // 重定向到帖子详情页面，这里根据实际情况修改重定向的路径
       return "redirect:/post/" + postId + "/" + uid;
    }

    @PostMapping("/reply/{postId}/{parentId}/{uid}")
    public String replyComment(@RequestParam String content,@PathVariable Long postId, @PathVariable Long parentId,@PathVariable Long uid, Model model, HttpSession session) {
        // 根据 uid 查找用户
        User user = userService.findUserByUid(uid);
        if (user == null) {
            // 处理未找到用户的情况，这里可以根据实际情况处理，比如跳转到错误页面或者其他逻辑
            return "redirect:/register";
        }

        Comment comment = new Comment(-1L, parentId, uid, content);
        commentService.saveComment(comment);
        Comment parent = commentService.getCommentById(parentId);
        parent.addChildren(parent);

        // 重定向到帖子详情页面，这里根据实际情况修改重定向的路径
       return "redirect:/post/" + postId + "/" + uid;
    }




}
